"""
Generate some multiplication examples to use for prompt engineering.
Use random A and B, where A is always larger than B.
Generate numbers between 0 and 99, inclusive.

E.g.
7 * 6 = 42
53 * 17 = 901
28 * 35 = 980
"""

import random

random.seed(13.13)


def generate_examples(n):
    output = ''
    for _ in range(n):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        if a < b:
            a, b = b, a  # A is always larger than B
        answer = a * b
        line = f'{a} * {b} = {answer}'
        output += line + r'\n'

    print(r'Multiply:\n' + output)


if __name__ == '__main__':
    generate_examples(4)
