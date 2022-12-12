"""
Generate some multiplication examples to use for prompt engineering.
Use random A and B.
Generate numbers between 0 and 99, inclusive.

E.g.
7 * 6 = 42
53 * 17 = 901
28 * 35 = 980

"""

import random

random.seed(66)

OUTPUT_FILE = 'data/prompt.csv'


def generate_examples_rand_sample(n):
    output = ''
    for _ in range(n):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        answer = a * b
        line = f'{a} * {b} = {answer}'
        output += line + r'\n'

    print(r'Multiply:\n' + output)


def generate_examples_spread_out_digits():
    """
    Evenly spread out the digits 0-10 into 3 groups of 2 multiplicads with 1 or 2 digits each.
    So that the multiplicands are diverse.
    """
    # First, randomly arrange the 10 digits.
    digits = list(range(10))
    random.shuffle(digits)

    # Then, partition them into 5 groups of 2 digits each.
    groups = []
    for i in range(0, 10, 2):
        groups.append(digits[i:i + 2])

    # Then, split 1 random group into 2 groups of 1 digit each.
    group_to_split = random.randint(0, 4)
    a, b = groups[group_to_split][0], groups[group_to_split][1]
    groups[group_to_split] = [a]
    groups.append([b])

    # Join the pairs of digits within each group into a single number.
    groups = [int(''.join(map(str, group))) for group in groups]

    # Randomly shuffle the groups.
    random.shuffle(groups)

    # Now, generate the prompt.
    examples = ''
    for i in range(0, len(groups), 2):
        a, b = groups[i], groups[i + 1]
        answer = a * b
        line = f'{a} * {b} = {answer}'
        examples += line + r'\n'
    print(r'Multiply:\n' + examples)

    # Write the examples to a file with headers a, b, completion
    with open(OUTPUT_FILE, 'w') as f:
        f.write('a,b,completion\n')
        for i in range(0, len(groups), 2):
            a, b = groups[i], groups[i + 1]
            answer = a * b
            line = f'{a},{b},{answer}'
            f.write(line + '\n')


if __name__ == '__main__':
    # generate_examples_rand_sample(3)
    generate_examples_spread_out_digits()
