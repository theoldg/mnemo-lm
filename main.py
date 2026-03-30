from typing import cast

from fire import Fire
from transformers import LogitsProcessorList

from mnemo_lm.digit_map import DIGIT_MAPS
from mnemo_lm.model import ModelAndTokenizer
from mnemo_lm.vocab_preprocessing import PreprocessedVocab
from mnemo_lm.logits_processor import ConstrainedMnemonicProcessor


def encode_digits(
    prompt: str,
    digits: list[int],
    model_and_tokenizer: ModelAndTokenizer,
    vocab: PreprocessedVocab,
    nudge: float,
    stop_nudge: float,
    generate_args: dict,
) -> list[str]:
    messages = [{"role": "user", "content": prompt}]
    text = model_and_tokenizer.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    text = cast(str, text)
    model_inputs = model_and_tokenizer.tokenizer([text], return_tensors="pt")

    processor = ConstrainedMnemonicProcessor(
        preprocessed_vocab=vocab,
        tokenizer=model_and_tokenizer.tokenizer,
        target=digits,
        prompt_size=model_inputs['input_ids'].shape[1],  # type: ignore
        nudge=nudge,
        stop_nudge=stop_nudge,
    )

    generated_ids = model_and_tokenizer.model.generate(
        **model_inputs,  # type: ignore
        **generate_args,
        logits_processor=LogitsProcessorList([processor]),
    )

    generated_strings = []
    for output_ids in generated_ids[:, len(model_inputs.input_ids[0]):]:
        generated_strings.append(
            model_and_tokenizer.tokenizer.decode(
                output_ids,
                skip_special_tokens=True
            ).strip("\n"), # type: ignore
        )

    return generated_strings


def main_interactive(lang: str = 'pl', model: str = 'qwen3-8b'):
    digit_map = DIGIT_MAPS[lang]
    model_and_tokenizer = ModelAndTokenizer.load(model_name=model)
    prepr_vocab = PreprocessedVocab.build(model_and_tokenizer.tokenizer, digit_map)

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
            model_and_tokenizer=model_and_tokenizer,
            vocab=prepr_vocab,
            nudge=2.5,
            stop_nudge=10,
            generate_args=dict(
                num_beams=20,
                no_repeat_ngram_size=3,
                early_stopping=True,
                max_new_tokens=10 * len(digits),
                num_return_sequences=10,
                # diversity_penalty=10.,
                # do_sample=True,
                # length_penalty=0.7,
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
