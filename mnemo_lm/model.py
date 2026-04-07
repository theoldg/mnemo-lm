"""Model short names and loading with tokenizer."""

from dataclasses import dataclass
from typing import Self

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TokenizersBackend,
    GenerationMixin,
)


MODEL_SHORT_NAMES = {
    'gemma4-4b': 'google/gemma-4-E4B-it',
    'gemma4-2b': 'google/gemma-4-E2B-it',
    'qwen3-8b': 'Qwen/Qwen3-8B',
    'qwen3-0.6b': 'Qwen/Qwen3-0.6B',
    'qwen3.5-9b': 'Qwen/Qwen3.5-9B',
}


@dataclass
class ModelAndTokenizer:
    model: GenerationMixin
    tokenizer: TokenizersBackend

    @classmethod
    def load(cls, model_name: str) -> Self:
        model_name = MODEL_SHORT_NAMES.get(model_name, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not hasattr(tokenizer, 'eos_token_id'):
            raise ValueError(f'The tokenizer for {model_name} does not have an `eos_token_id`.')
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        return cls(model=model, tokenizer=tokenizer)
