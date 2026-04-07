from fire import Fire


def main(
    lang: str = "pl",
    model: str = "gemma4-4b",
    nudge: float = 3.5,
    stop_nudge: float = 15.0,
    num_beams: int = 20,
    num_return_sequences: int = 10,
    max_tokens_per_digit: int = 10,
):
    """
    Interactive generator for LLM-powered Major System mnemonics.

    Args:
        lang: Language for the digit mapping (e.g., 'pl', 'en').
        model: Model short name or HuggingFace ID (e.g., 'qwen3-8b', 'qwen3-0.6b').
        nudge: Logit multiplier to boost the probabilities of matching tokens.
        stop_nudge: Logit multiplier to boost the EOS token when generation is complete.
        num_beams: Number of beams to use during beam search generation.
        num_return_sequences: Number of distinct mnemonic phrases to generate.
        max_tokens_per_digit: Force stops generation after the total budget is exhausted.
    """

    # Only import stuff if actually executing (not in the case of --help).
    from mnemo_lm.encode import encode_interactive

    encode_interactive(
        lang=lang,
        model=model,
        nudge=nudge,
        stop_nudge=stop_nudge,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_tokens_per_digit=max_tokens_per_digit,
    )


if __name__ == "__main__":
    Fire(main)
