"""
Generates the prompts and collects the model's responses.

2-shot 2-digit positive integer multiplication evaluated on GPT-3.
We always list the larger number first.
"""

from abc import ABC, abstractmethod


class Prompter(ABC):
    """Given numbers to multiply, format them into a prompt."""

    @abstractmethod
    def __call__(self, a: int, b: int) -> str:
        raise NotImplementedError


class TwoShotPrompter(Prompter):
    """2-shot example prompting to help our model learn the format."""

    def __init__(self, examples: str = ''):
        if examples == '':
            self.examples = '65 * 44 = 2860\n98 * 23 = 2254\n'  # Randomly chosen default
        else:
            self.examples = examples

    def __call__(self, a: int, b: int) -> str:
        return f'{self.examples}{str(a)} * {str(b)} ='


class Answerer(ABC):
    """Answers multiplication prompts."""

    @abstractmethod
    def __call__(self, prompts: list[str]) -> list[int | None]:
        """
        Answer a list of prompt strings, returning a list of answers to each prompt as ints.

        Returns None if no valid answer could be extracted for a given prompt.
        """
        raise NotImplementedError


class StubAnswerer(Answerer):
    """Output canned answers for testing."""

    def __call__(self, prompts: list[str]) -> list[int]:
        return [66] * len(prompts)


if __name__ == '__main__':
    # Generate numbers and prompts
    multiplicand_tuples = []
    prompts = []
    prompter = TwoShotPrompter()

    for a in range(0, 100):
        for b in range(0, a + 1):  # a >= b
            multiplicand_tuples.append((a, b))
            prompts.append(prompter(a, b))

    # Generate some answers
    answerer = StubAnswerer()
    answers = answerer(prompts)

    # Print the results
    for multiplicand_tuple, answer in zip(multiplicand_tuples, answers):
        a, b = multiplicand_tuple
        print(f'{a} * {b} = {answer}')

    num_answered = len([answer for answer in answers if answer is not None])
    print(f'Generated {len(prompts)} prompts and successfully got {num_answered} answers.')
