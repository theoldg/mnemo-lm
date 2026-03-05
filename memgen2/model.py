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
    'qwen-8b': 'Qwen/Qwen3-8B',
    'qwen-0.6b': 'Qwen/Qwen3-0.6B',
}


@dataclass
class ModelAndTokenizer:
    model: GenerationMixin
    tokenizer: TokenizersBackend

    @classmethod
    def load(cls, model_name: str) -> Self:
        model_name = MODEL_SHORT_NAMES.get(model_name, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        return cls(model=model, tokenizer=tokenizer)
