from fire import Fire

from mnemo_lm.digit_map import DigitMap, DIGIT_MAPS


def decode_and_format(x: str, digit_map: DigitMap):
    return ''.join(map(str, digit_map.apply(x)))


def main(query: str | None = None, lang: str = 'pl'):
    """
    Args:
        query: Optional. The string to decode. If not provided, 
            an interactive loop is started.
        lang: Language for the digit mapping.
    """
    digit_map = DIGIT_MAPS[lang]
    if query:
        print(decode_and_format(query, digit_map))
        return
    
    while True:
        try:
            prompt = 'Phrase to decode: '
            query = input(prompt)
            print(
                f'{"Decoded:":<{len(prompt) - 1}}',
                decode_and_format(query, digit_map),
            )
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    Fire(main)
