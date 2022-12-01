"""
Generates the prompts and collects the model's responses.

2-shot 2-digit positive integer multiplication evaluated on GPT-3.
We always list the larger number first.
"""

import os
import csv
from abc import ABC, abstractmethod

from tqdm import tqdm
import openai
from openai.error import RateLimitError
import backoff
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

START_MULTIPLICAND = 0
STOP_MULTIPLICAND = 100
OUTPUT_FILENAME = f'./data/multiplication_data_gptneo1,3_v2_{START_MULTIPLICAND}_{STOP_MULTIPLICAND}.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Prompter(ABC):
    """Given numbers to multiply, format them into a prompt."""

    @abstractmethod
    def __call__(self, a: int, b: int) -> str:
        raise NotImplementedError


class TwoShotPrompter(Prompter):
    """2-shot example prompting to help our model learn the format."""

    def __init__(self, examples: str = ''):
        if examples == '':
            self.examples = '7 * 6 = 42\n65 * 44 = 2860\n98 * 23 = 2254\n'  # Randomly chosen default
        else:
            self.examples = examples

    def __call__(self, a: int, b: int) -> str:
        return f'{self.examples}{str(a)} * {str(b)} ='


class Answerer(ABC):
    """Answers multiplication prompts."""

    @abstractmethod
    def __call__(self, prompts: list[str]) -> list[str | None]:
        """
        Answer a list of prompt strings, returning a list of answers to each prompt as ints.

        Returns None if no valid answer could be extracted for a given prompt.
        """
        raise NotImplementedError


class StubAnswerer(Answerer):
    """Output canned answers for testing."""

    def __call__(self, prompts: list[str]) -> list[str]:
        return ['66'] * len(prompts)


class GPT3APIAnswerer(Answerer):
    def __init__(self, model_name='text-davinci-003'):
        self.model_name = model_name

    @backoff.on_exception(backoff.expo, RateLimitError)
    def completions_with_backoff(self, **kwargs):
        """https://help.openai.com/en/articles/5955604-how-can-i-solve-429-too-many-requests-errors"""
        response = openai.Completion.create(**kwargs)
        return response

    def __call__(self, prompts: list[str]) -> list[str | None]:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        output = []
        try:
            for i in tqdm(range(0, len(prompts), 20)):
                batch = prompts[i:i+20]
                completion = openai.Completion.create(
                    model=self.model_name,
                    prompt=batch,
                    max_tokens=6,
                    temperature=0,
                    stop='\n'
                )
                output += [choice.text.strip() for choice in completion['choices']]  # type: ignore
        except Exception as e:
            print(e)
        return output


class HFTransformersAnswerer(Answerer):
    def __init__(self, model_name='gpt2', batch_size=128):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eos_token = self.tokenizer('\n').input_ids[0]
        self.tokenizer.pad_token = self.eos_token

    def __call__(self, prompts: list[str]) -> list[str | None]:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        output = []
        for i in tqdm(range(0, len(prompts), self.batch_size)):
            batch = prompts[i:i+self.batch_size]
            tokenized = self.tokenizer(batch, padding=True, return_tensors='pt')
            input_ids = tokenized.input_ids.to(DEVICE)
            attention_mask = tokenized.attention_mask.to(DEVICE)
            generated = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                temperature=0.0,
                pad_token_id=self.eos_token,
                eos_token_id=self.eos_token,
            )
            only_new_tokens = generated[:, tokenized.input_ids[0].shape[0]:]
            decoded_completions = self.tokenizer.batch_decode(only_new_tokens)
            output += [completion.strip() for completion in decoded_completions]
        return output


if __name__ == '__main__':
    # Generate numbers and prompts
    multiplicand_tuples = []
    prompts = []
    prompter = TwoShotPrompter()

    for a in range(START_MULTIPLICAND, STOP_MULTIPLICAND):
        for b in range(0, a + 1):  # a >= b
            multiplicand_tuples.append((a, b))
            prompts.append(prompter(a, b))

    # Choose a model with which to answer
    # answerer = StubAnswerer()
    # answerer = GPT3APIAnswerer()
    # answerer = HFTransformersAnswerer()
    # answerer = HFTransformersAnswerer('EleutherAI/gpt-j-6B')
    answerer = HFTransformersAnswerer('EleutherAI/gpt-neo-1.3B', batch_size=16)

    # Generate some answers
    answers = answerer(prompts)

    # Print the results
    for multiplicand_tuple, answer in zip(multiplicand_tuples, answers):
        a, b = multiplicand_tuple
        print(f'{a} * {b} = {answer}')

    num_answered = len([answer for answer in answers if answer is not None])
    print(f'Generated {len(prompts)} prompts and successfully got {num_answered} answers.')

    # Write results as CSV
    with open(OUTPUT_FILENAME, 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(['a', 'b', 'completion'])
        for multiplicand_tuple, answer in zip(multiplicand_tuples, answers):
            writer.writerow([multiplicand_tuple[0], multiplicand_tuple[1], answer])
