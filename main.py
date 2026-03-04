from collections import defaultdict
from dataclasses import dataclass
from typing import Self, cast
import re

from fire import Fire
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    TokenizersBackend,
    GenerationMixin,
)
import torch


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
    
    @classmethod
    def new(cls):
        return cls()

    def add(self, lst: list[int], id: int):
        if not lst:
            self.tokens.append(id)
            return
        if lst[0] not in self.branches:
            self.branches[lst[0]] = self.new()
        self.branches[lst[0]].add(lst[1:], id)

    def build(self) -> PrefixTree:
        return PrefixTree(
            tokens=torch.tensor(self.tokens, dtype=torch.long),
            branches={k: v.build() for k, v in self.branches.items()}
        )


@dataclass
class PreprocessedVocab:
    digits: list[list[int]]
    strings: list[str]
    tree: PrefixTree
    startswith: dict[str, torch.Tensor]
    digit_map: DigitMap

    def __len__(self):
        return len(self.digits)


def preprocess_vocab(
    tokenizer: TokenizersBackend,
    digit_map: DigitMap,
) -> PreprocessedVocab:
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
        tree.add(toks, i)

    # Track which tokens start with a second letter of a digraph.
    digraph_suffixes = sum(digit_map.digraph_map.values(), [])
    startswith: dict[str, torch.Tensor] = {}
    for letter in digraph_suffixes:
        lst = []
        for i, s in enumerate(strings):
            if s.startswith(letter):
                lst.append(i)
        startswith[letter] = torch.tensor(lst, dtype=torch.long)

    return PreprocessedVocab(
        digits=sequences,
        strings=strings,
        tree=tree.build(),
        startswith=startswith,
        digit_map=digit_map,
    )


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


@dataclass
class ModelAndTokenizer:
    model: GenerationMixin
    tokenizer: TokenizersBackend


def load_model_and_tokenizer(big: bool) -> ModelAndTokenizer:
    if big:
        model_name = "Qwen/Qwen3-8B"
    else:
        model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return ModelAndTokenizer(model=model, tokenizer=tokenizer)


def encode_digits(
    prompt: str,
    digits: list[int],
    mnt: ModelAndTokenizer,
    vocab: PreprocessedVocab,
    nudge: float,
    stop_nudge: float,
    generate_args: dict,
) -> list[str]:
    messages = [{"role": "user", "content": prompt}]
    text = mnt.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    text = cast(str, text)
    model_inputs = mnt.tokenizer([text], return_tensors="pt")

    processor = ConstrainedMnemonicProcessor(
        preprocessed_vocab=vocab,
        tokenizer=mnt.tokenizer,
        target=digits,
        prompt_size=model_inputs['input_ids'].shape[1],  # type: ignore
        nudge=nudge,
        stop_nudge=stop_nudge,
    )

    generated_ids = mnt.model.generate(
        **model_inputs,  # type: ignore
        **generate_args,
        logits_processor=LogitsProcessorList([processor]),
    )

    generated_strings = []
    for output_ids in generated_ids[:, len(model_inputs.input_ids[0]):]:
        generated_strings.append(
            mnt.tokenizer.decode(
                output_ids,
                skip_special_tokens=True
            ).strip("\n"), # type: ignore
        )

    return generated_strings


def main_interactive(lang: str = 'pl', big: bool = True):
    digit_map = DIGIT_MAPS[lang]
    mnt = load_model_and_tokenizer(big=big)
    prepr_vocab = preprocess_vocab(mnt.tokenizer, digit_map)

    while True:
        input_string = input('Digits: ')
        try:
            digits = list(map(int, str(input_string)))
            assert digits
        except:
            print('Invalid input')
            continue
        prompt = input('Prompt (skip for default): ') or digit_map.default_prompt
    
        encoded_strings = encode_digits(
            prompt,
            digits,
            mnt=mnt,
            vocab=prepr_vocab,
            nudge=2.5,
            stop_nudge=10,
            generate_args=dict(
                num_beams=20,
                no_repeat_ngram_size=3,
                # diversity_penalty=10.,
                # do_sample=True,
                early_stopping=True,
                # length_penalty=0.7,
                max_new_tokens=10 * len(digits),
                num_return_sequences=10,
            ),
        )

        for encoded in encoded_strings:
            print('+' * 50)
            decoded = digit_map.apply(encoded)
            if decoded != digits:
                print('[Error]. Back-decoded:', ''.join(map(str, decoded)))
            print(encoded)
        print('-' * 50 + '\n')


if __name__ == '__main__':
    Fire(main_interactive)
