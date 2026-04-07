"""Main logic for constrained generation."""

import torch
from torch import LongTensor, FloatTensor
from transformers import LogitsProcessor, TokenizersBackend
from tqdm.auto import tqdm
from typing import cast, no_type_check

from mnemo_lm.vocab_preprocessing import PreprocessedVocab


class ConstrainedMnemonicProcessor(LogitsProcessor):
    def __init__(
        self,
        preprocessed_vocab: PreprocessedVocab,
        tokenizer: TokenizersBackend,
        target: list[int],
        prompt_size: int,
        nudge: float,
        stop_nudge: float,
        progress: bool = True,
    ):
        super().__init__()
        self.preprocessed_vocab = preprocessed_vocab
        self.tokenizer = tokenizer
        self.target = target
        self.prompt_size = prompt_size
        self.nudge = nudge
        self.stop_nudge = stop_nudge
        if progress:
            self.pbar = tqdm(desc="Searching...")
        else:
            self.pbar = None

    def call_single(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        """Return updated next token logits.
        
        Suppose we already generated tokens [..., Tx, Ty, Tz], which cover some
        of the target digits (`self.target`), so that the remaining digits to
        encode are [Di, Dj, Dk, ...].

        This function does the following:
            - Forbid any token which maps to undesired digits by replacing its
              logit with -inf.
            - Forbid any token which creates a digraph with the previous token,
              i.e. any Tw such that digits([Tz] + [Tw]) != digits([Tz]) + digits([Tw]).
            - Allow any neutral token, i.e. any Tw where digits([Tw]) = [].
            - Boost any token which generates desired digits:
              If digits([Tw]) = [Di], add `self.nudge` to the logit,
              if digits([Tw]) = [Di, Dj], add `2 * self.nudge`, etc.
            - If there are no more digits to encode, boost the probability of
              generating [EOS] by adding `stop_nudge` to the corresponding logit.

        Args:
            input_ids: Current generation sequence, 1D tensor with token IDs.
            scores: Next token logits from the model.

        Returns:
            Modified scores (logits).
        """
        current_digits = []
        for tok in input_ids[self.prompt_size :]:
            if tok >= len(self.preprocessed_vocab):
                continue
            current_digits.extend(self.preprocessed_vocab.digits[tok])

        remaining_digits = self.target[len(current_digits) :]

        # Mask everything by default.
        mask = torch.zeros_like(scores, dtype=torch.bool)

        if self.pbar is not None:
            self.pbar.set_description(
                f"Generated: {len(input_ids) - self.prompt_size}. "
                f"Remaining digits: {len(remaining_digits)}"
            )
            self.pbar.update()

        if not remaining_digits:
            # Unmask and nudge the EOS token.
            eos_id = cast(int, self.tokenizer.eos_token_id)
            scores[eos_id] += self.stop_nudge
            mask[eos_id] = True

        # Unmask neutral tokens.
        tree = self.preprocessed_vocab.tree
        if len(tree.tokens) > 0:
            mask[tree.tokens] = True
        # Unmask and nudge useful tokens, as deep as the prefix tree goes.
        for i, digit in enumerate(remaining_digits):
            tree = tree.branches.get(digit)
            if not tree:
                break
            if len(tree.tokens) > 0:
                scores[tree.tokens] += self.nudge * (i + 1)
                mask[tree.tokens] = True

        # Re-mask anything with a forbidden prefix.
        last_token = self.preprocessed_vocab.strings[input_ids[-1]]
        if last_token:
            last_letter = last_token[-1]
            digraph_map = self.preprocessed_vocab.digit_map.digraph_map
            forbidden_prefixes = digraph_map.get(last_letter, [])
            for prefix in forbidden_prefixes:
                mask[self.preprocessed_vocab.startswith[prefix]] = False

        # Apply mask.
        scores[~mask] = -torch.inf
        return scores

    @no_type_check
    def __call__(
        self,
        input_ids: LongTensor,
        scores: torch.FloatTensor,
    ) -> FloatTensor:
        return torch.stack(
            [self.call_single(i, s) for i, s in zip(input_ids, scores)]
        )
