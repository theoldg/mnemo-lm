# mnemo_lm

**LLM-powered mnemonic generation** 

`mnemo_lm` uses constrained decoding to force Large Language Models 
(like Qwen) to generate coherent phrases that encode specific sequences
of digits using the Mnemonic Major System.

## Installation
[Get `uv`](https://docs.astral.sh/uv/getting-started/installation/) 
if you don't have it, then run `uv sync`.

## Usage
Run the interactive generator:
```bash
# Defaults to Polish and qwen-8b
uv run main.py

# Specify language or model
uv run main.py --lang en --model qwen-0.6b
```

## Example: Digit Mapping for Polish
| Digit | Letters |
| :--- | :--- |
| 1 | T, D |
| 2 | N, Ń |
| 3 | M |
| 4 | R |
| 5 | L |
| 6 | SZ, CZ, DŻ, DZ, DŹ, RZ, Ż |
| 7 | K, G |
| 8 | F, W |
| 9 | P, B |
| 0 | S, Z, Ś, Ź |


## How it Works

At each decoding step, the model's output probabilities are modified.
Tokens which would map to incorrect digits are masked, while tokens that
map to desired digits get a boosted probability. See the logit processor 
[implementation](mnemo_lm/logits_processor.py) for more.
