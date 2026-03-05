"""Preprocessing of model vocab to optimize on-the-fly constrained generation."""

from dataclasses import dataclass
from typing import cast, Self

import torch
from transformers import TokenizersBackend
from tqdm.auto import tqdm

from memgen2.prefix_tree import PrefixTree, PrefixTreeBuilder
from memgen2.digit_map import DigitMap


@dataclass
class PreprocessedVocab:
    digits: list[list[int]]
    strings: list[str]
    tree: PrefixTree
    startswith: dict[str, torch.Tensor]
    digit_map: DigitMap

    def __len__(self):
        return len(self.digits)

    @classmethod
    def build(
        cls,
        tokenizer: TokenizersBackend,
        digit_map: DigitMap,
    ) -> Self:
        vocab_size = len(tokenizer.vocab)
        sequences: list = [None for _ in range(vocab_size)]
        strings: list = [None for _ in range(vocab_size)]
        for s, index in tqdm(tokenizer.vocab.items()):
            s = tokenizer.decode(index)
            s = cast(str, s)
            digits = digit_map.apply(s)
            sequences[index] = digits
            strings[index] = s.upper()

        tree = PrefixTreeBuilder()
        for i, toks in enumerate(sequences):
            tree.insert(toks, i)

        # Track which tokens start with a second letter of a digraph.
        digraph_suffixes = sum(digit_map.digraph_map.values(), [])
        startswith: dict[str, torch.Tensor] = {}
        for letter in digraph_suffixes:
            lst = []
            for i, s in enumerate(strings):
                if s.startswith(letter):
                    lst.append(i)
            startswith[letter] = torch.tensor(lst, dtype=torch.long)

        return cls(
            digits=sequences,
            strings=strings,
            tree=tree.build(),
            startswith=startswith,
            digit_map=digit_map,
        )
