"""Main logic for contrained generation."""

import torch
from transformers import LogitsProcessor, TokenizersBackend
from tqdm.auto import tqdm

from memgen2.vocab_preprocessing import PreprocessedVocab


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
            self.pbar = tqdm(desc='Searching...')
        else:
            self.pbar = None

    def call_single(self, input_ids, scores):
        current_digits = []
        for tok in input_ids[self.prompt_size:]:
            if tok >= len(self.preprocessed_vocab):
                continue
            current_digits.extend(self.preprocessed_vocab.digits[tok])

        mask = torch.zeros_like(scores, dtype=torch.bool)

        remaining_digits = self.target[len(current_digits):]
        
        if self.pbar is not None:
            self.pbar.set_description(
                f'Generated: {len(input_ids) - self.prompt_size}. '
                f'Remaining digits: {len(remaining_digits)}'
            )
            self.pbar.update()

        # Unmask and nudge the EOS token.
        if not remaining_digits:
            eos_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            scores[eos_id] += self.stop_nudge
            mask[eos_id] = 1

        # Unmask and nudge relevant tokens.
        for i in range(len(remaining_digits) + 1):
            results = self.preprocessed_vocab.tree.get(remaining_digits[:i])
            if len(results) == 0:
                continue
            scores[results] += self.nudge * i
            mask[results] = 1

        # Re-mask anything with a forbidden prefix.
        last_token = self.preprocessed_vocab.strings[input_ids[-1]]
        if last_token:
            last_letter = last_token[-1]
            digraph_map = self.preprocessed_vocab.digit_map.digraph_map
            forbidden_prefixes = digraph_map.get(last_letter, [])
            for prefix in forbidden_prefixes:
                mask[self.preprocessed_vocab.startswith[prefix]] = 0
        
        # Apply mask.
        scores[~mask] = -torch.inf
        return scores

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return torch.stack([
            self.call_single(i, s)
            for i, s in zip(input_ids, scores)
        ])  # type: ignore

