"""Main logic for constrained generation."""

import torch
from transformers import LogitsProcessor, TokenizersBackend
from tqdm.auto import tqdm
from typing import cast

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

    def call_single(self, input_ids, scores):
        current_digits = []
        for tok in input_ids[self.prompt_size :]:
            if tok >= len(self.preprocessed_vocab):
                continue
            current_digits.extend(self.preprocessed_vocab.digits[tok])

        mask = torch.zeros_like(scores, dtype=torch.bool)

        remaining_digits = self.target[len(current_digits) :]

        if self.pbar is not None:
            self.pbar.set_description(
                f"Generated: {len(input_ids) - self.prompt_size}. "
                f"Remaining digits: {len(remaining_digits)}"
            )
            self.pbar.update()

        # Unmask and nudge the EOS token.
        if not remaining_digits:
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

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return torch.stack(
            [self.call_single(i, s) for i, s in zip(input_ids, scores)]
        )  # type: ignore
