"""Mapping strings to sequences of digits in different languages."""

from collections import defaultdict
import re


class DigitMap:
    map: dict[str, int]
    pattern: re.Pattern
    digraphs: dict[str, str]

    def __init__(
        self,
        default_prompt: str,
        map: dict[str, int],
    ):
        self.default_prompt = default_prompt
        self.map = map
        self.pattern = re.compile('|'.join(
            # Longest first!
            sorted(map.keys(), key=len, reverse=True))
        )
        
        self.digraph_map = defaultdict(list)
        for key in map:
            if len(key) == 2:
                x, y = key
                self.digraph_map[x].append(y)
            elif len(key) >= 3:
                raise ValueError('Trigraphs are not supported yet!')

    def apply(self, s: str) -> list[int]:
        matches = self.pattern.findall(s.upper())
        return [self.map[m] for m in matches]


PL_MAP = DigitMap(
    default_prompt="Wymyśl krótkie zdanie brzmiące jak Polskie przysłowie",
    map={
        'T': 1, 'D': 1,
        'N': 2, 'Ń': 2,
        'M': 3,
        'R': 4,
        'L': 5,
        'SZ': 6, 'CZ': 6, 'DŻ': 6, 'DZ': 6, 'DŹ': 6, 'RZ': 6, 'Ż': 6,
        'K': 7, 'G': 7,
        'F': 8, 'W': 8,
        'P': 9, 'B': 9,
        'Z': 0, 'S': 0, 'Ź': 0, 'Ś': 0,
    },
)


# Dummy map, no phoneme handling, written by G.
EN_MAP = DigitMap(
    default_prompt="Write a snappy made up proverb.",
    map={
        # --- Digraphs ---
        'TH': 1,
        'SH': 6, 'CH': 6,
        'CK': 7,
        'PH': 8,
        
        # --- Double Consonants (to prevent double-counting) ---
        'TT': 1, 'DD': 1,
        'NN': 2,
        'MM': 3,
        'RR': 4,
        'LL': 5,
        'GG': 7, 'KK': 7, 'CC': 7,
        'FF': 8, 'VV': 8,
        'PP': 9, 'BB': 9,
        'SS': 0, 'ZZ': 0,

        # --- Single Consonants ---
        'T': 1, 'D': 1,
        'N': 2,
        'M': 3,
        'R': 4,
        'L': 5,
        'J': 6,
        'K': 7, 'G': 7, 'Q': 7, 'C': 7,
        'F': 8, 'V': 8,
        'P': 9, 'B': 9,
        'S': 0, 'Z': 0,
    },
)


DIGIT_MAPS = {
    'pl': PL_MAP,
    'en': EN_MAP,
}
