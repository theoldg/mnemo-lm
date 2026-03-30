# mnemo_lm

**LLM-powered mnemonic generation** 

`mnemo_lm` uses constrained decoding to force Large Language Models 
(like Qwen) to generate coherent phrases that encode specific sequences
of digits using the Mnemonic Major System.

## Setup
[Get `uv`](https://docs.astral.sh/uv/getting-started/installation/) and run `uv sync`.

## Usage
Run the interactive generator:
```bash
# Defaults to Polish and qwen3-8b
uv run main.py

# Specify language or model
uv run main.py --lang en --model qwen3-0.6b
```

Then, type the digits you want to convert and, optionally, a prompt for the LLM.

You can see the full list of flags using `uv run main.py --help`.

> Note: English language support is currently limited. It uses an imperfect letter-based mapping
> which you can find in [the code](mnemo_lm/digit_map.py).

### Example

Suppose you want to remember the digits `293848`. Run the script and type them in:

```
$ uv run main.py --lang pl --model qwen3-8b
[...]
Digits: 293848
Prompt (skip for default): 
Generated: 25. Remaining digits: 0: : 520it [00:45, 11.50it/s]
++++++++++++++++++++++++++++++++++++++++++++++++++
Nie pojmuję, co wyrwało.
++++++++++++++++++++++++++++++++++++++++++++++++++
Na ciepło mówiąc – ręce wajchają.
++++++++++++++++++++++++++++++++++++++++++++++++++
Na ciepło mówiąc, ręce wije.
[...]
```

Choose your favourite mnemonic sentence among the generated suggestions. Once you remember the mnemonic, you can recover the number by mapping the special letters back to digits.

"`N`ie `P`oj`M`uję, co `W`y`RW`ało" → 2 9 3 8 4 8

## Digit Mapping for Polish
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

> Note: Vowels like A, E, I, O, U, Y, and letters like C, H, J are ignored and act as "filler" to help create words.


## How it Works

At each decoding step, the model's output probabilities are modified.
Tokens which would map to incorrect digits are masked, while tokens that
map to desired digits get a boosted probability. See the logit processor 
[implementation](mnemo_lm/logits_processor.py) for more.

