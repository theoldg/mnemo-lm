"""Prefix tree whose leaves are index tensors."""

from dataclasses import dataclass
from typing import Self

import torch


@dataclass
class PrefixTree:
    tokens: torch.Tensor
    branches: dict[int, Self]

    def get(self, lst: list[int]) -> torch.Tensor:
        if not lst:
            return self.tokens
        if lst[0] not in self.branches:
            return torch.tensor([])
        return self.branches[lst[0]].get(lst[1:])


class PrefixTreeBuilder:
    def __init__(self):
        self.tokens: list = []
        self.branches: dict[int, Self] = {}
    
    def insert(self, lst: list[int], id: int):
        if not lst:
            self.tokens.append(id)
            return
        if lst[0] not in self.branches:
            self.branches[lst[0]] = type(self)()
        self.branches[lst[0]].insert(lst[1:], id)

    def build(self) -> PrefixTree:
        return PrefixTree(
            tokens=torch.tensor(self.tokens, dtype=torch.long),
            branches={k: v.build() for k, v in self.branches.items()}
        )
