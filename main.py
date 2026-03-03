from dataclasses import dataclass
import typing
from typing import Self

from fire import Fire
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import torch

MAP = {
    'T': 1, 'D': 1,
    'N': 2, 'Ń': 2,
    'M': 3,
    'R': 4,
    'L': 5,
    'SZ': 6, 'CZ': 6, 'Ż': 6, 'DŻ': 6, 'DZ': 6, 'DŹ': 6,
    'K': 7, 'G': 7,
    'F': 8, 'W': 8,
    'P': 9, 'B': 9,
    'Z': 0, 'S': 0, 'Ź': 0, 'Ś': 0,
    'CH': None,
}


def string_to_digits(s: str):
    pos = 0
    s = s.upper()
    digits = []
    while pos < len(s):
        substring = s[pos: pos + 2]
        if substring in MAP:
            digits.append(MAP[substring])
            pos += 2
            continue
        substring = s[pos: pos + 1]
        if substring in MAP:
            digits.append(MAP[substring])
            pos += 1
            continue
        pos += 1
    return [d for d in digits if d is not None]


@dataclass
class ModelAndTokenizer:
    model: Qwen3ForCausalLM
    tokenizer: Qwen2Tokenizer


class PrefixTree:
    def __init__(self):
        self.tokens: list | torch.Tensor = []
        self.branches: dict[int, Self] = {}
        

def add_to_tree(t: PrefixTree, lst: list[int], id: int):
    if not lst:
        assert isinstance(t.tokens, list)
        t.tokens.append(id)
        return

    if lst[0] not in t.branches:
        t.branches[lst[0]] = PrefixTree()
    
    add_to_tree(t.branches[lst[0]], lst[1:], id)


def tensorize_tree(t: PrefixTree):
    t.tokens = torch.tensor(t.tokens, dtype=torch.long)
    for subtree in t.branches.values():
        tensorize_tree(subtree)


def get_from_tree(t: PrefixTree, lst: list[int]):
    if not lst:
        return t.tokens

    if lst[0] not in t.branches:
        return []

    return get_from_tree(t.branches[lst[0]], lst[1:])


@dataclass
class PreprocessedVocab:
    digits: list[list[int]]
    strings: list[str]
    tree: PrefixTree
    startswith: dict[str, int]

    def __len__(self):
        return len(self.digits)


class MemGenProcessor(LogitsProcessor):
    def __init__(
        self,
        preprocessed_vocab: PreprocessedVocab,
        tokenizer: Qwen2Tokenizer,
        target: list[int],
        prompt_size: int,
        nudge: float,
        stop_nudge: float,
    ):
        super().__init__()
        self.preprocessed_vocab = preprocessed_vocab
        self.tokenizer = tokenizer
        self.target = target
        self.prompt_size = prompt_size
        self.nudge = nudge
        self.stop_nudge = stop_nudge
    

    def call_single(self, input_ids, scores):
        current_digits = []
        for tok in input_ids[self.prompt_size:]:
            if tok >= len(self.preprocessed_vocab):
                continue
            current_digits.extend(self.preprocessed_vocab.digits[tok])

        mask = torch.zeros_like(scores, dtype=torch.bool)

        remaining_digits = self.target[len(current_digits):]

        # Unmask and nudge the EOS token.
        if not remaining_digits:
            eos_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            scores[eos_id] += self.stop_nudge
            mask[eos_id] = 1

        # Unmask and nudge relevant tokens.
        for i in range(len(remaining_digits) + 1):
            results = get_from_tree(self.preprocessed_vocab.tree, remaining_digits[:i])
            if len(results) == 0:
                continue
            scores[results] += self.nudge * i
            mask[results] = 1

        # Re-mask anything with a forbidden prefix.
        last_letter = self.preprocessed_vocab.strings[input_ids[-1]][-1]
        if (last_letter == 'S') or (last_letter == 'C') :
            forbidden_prefixes = 'Z'
        elif last_letter == 'D':
            forbidden_prefixes = 'ŻZŹ'
        else:
            forbidden_prefixes = ''
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


@typing.no_type_check
def load_model_and_tokenizer() -> ModelAndTokenizer:
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return ModelAndTokenizer(model=model, tokenizer=tokenizer)


def preprocess_vocab(tokenizer: Qwen2Tokenizer) -> PreprocessedVocab:
    vocab_size = len(tokenizer.vocab)
    sequences: list = [None for _ in range(vocab_size)]
    strings: list = [None for _ in range(vocab_size)]
    for s, index in tqdm(tokenizer.vocab.items()):
        s = tokenizer.decode(index)
        digits = string_to_digits(s)  # type: ignore
        sequences[index] = digits
        strings[index] = s.upper()  # type: ignore

    tree = PrefixTree()
    for i, toks in enumerate(sequences):
        add_to_tree(tree, toks, i)
    tensorize_tree(tree)

    startswith = {}
    for letter in 'ZŻŹ':
        lst = []
        for i, s in enumerate(strings):
            if s.startswith(letter):
                lst.append(i)
        startswith[letter] = torch.tensor(lst, dtype=torch.long)

    return PreprocessedVocab(
        digits=sequences,
        strings=strings,
        tree=tree,
        startswith=startswith,
    )


def encode_digits(
    digits: list[int],
    mnt: ModelAndTokenizer,
    vocab: PreprocessedVocab,
    nudge: float,
    stop_nudge: float,
    generate_args: dict,
) -> list[str]:
    prompt = "Napisz przykładowe zdanie po Polsku,"

    messages = [{"role": "user", "content": prompt}]
    text: str = mnt.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )  # type: ignore
    model_inputs = mnt.tokenizer([text], return_tensors="pt").to(mnt.model.device)

    processor = MemGenProcessor(
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
                skip_special_tokens=True).strip("\n"), # type: ignore
        )

    return generated_strings


def main_interactive():
    mnt = load_model_and_tokenizer()
    prepr_vocab = preprocess_vocab(mnt.tokenizer)

    while True:
        input_string = input('Type the digits you want to encode: ')
        try:
            digits = list(map(int, str(input_string)))
            assert digits
        except:
            print('Invalid input')
            continue
    
        encoded_strings = encode_digits(
            digits,
            mnt=mnt,
            vocab=prepr_vocab,
            nudge=3,
            stop_nudge=10,
            generate_args=dict(
                num_beams=20,
                diversity_penalty=10.,
                do_sample=True,
                early_stopping=True,
                length_penalty=0.7,
                max_new_tokens=10 * len(digits),
                num_return_sequences=10,
            ),
        )

        for encoded in encoded_strings:
            print('+' * 50)
            print(encoded)
            decoded = string_to_digits(encoded)
            if decoded != digits:
                print('Error. Back-decoded:', ''.join(map(str, decoded)))
        print('-' * 50 + '\n')


if __name__ == '__main__':
    Fire(main_interactive)
