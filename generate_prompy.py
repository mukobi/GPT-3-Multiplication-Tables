"""
Generate some multiplication examples to use for prompt engineering.
Use random A and B, where A is always larger than B.
Generate numbers between 0 and 1000, inclusive.

E.g.
7 * 6 = 42
531 * 127 = 67437
128 * 35 = 4480
"""

import random

random.seed(66)


def generate_examples(n):
    output = ''
    for _ in range(n):
        a = random.randint(0, 1000)
        b = random.randint(0, 1000)
        if a < b:
            a, b = b, a  # A is always larger than B
        answer = a * b
        line = f'{a} * {b} = {answer}'
        output += line + r'\n'

    print(r'Multiply:\n' + output)


if __name__ == '__main__':
    generate_examples(10)
