import os
import csv
import re
import matplotlib.pyplot as plt

FILE_SUBSTRING_TO_NAME = {
    'GPT3-text-ada-001': 'GPT-3 Ada (350M)',
    'GPT3-text-babbage-001': 'GPT-3 Babbage (1.3B)',
    'GPT3-text-curie-001': 'GPT-3 Curie (6.7B)',
    'GPT3-text-davinci-003': 'GPT-3 Davinci (175B)',
    'EleutherAI-gpt-neo-1.3B': 'GPT-Neo-1.3B (1.3B)',
    'gpt2': 'GPT-2 (117M)',
    'Stub': 'Stub',
    'prompt': 'Prompt',
}


def plot_heatmap(filename):
    # Initialize empty lists for the a, b, and completion values
    a_values = []
    b_values = []
    completion_values = []

    # Convert filename to model name
    model_name = 'Unknown'
    for substring, name in FILE_SUBSTRING_TO_NAME.items():
        if substring in filename:
            model_name = name
            break

    # Initialize empty grid of zeros (100x100)
    correctness_grid = [[0 for i in range(100)] for j in range(100)]
    digits_off_grid = [[-1 for i in range(100)] for j in range(100)]

    # Open the CSV file and read the values into the lists
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the first row of the CSV file (the column names)
        for row in reader:
            a = int(row[0])
            b = int(row[1])
            completion_numeric = re.sub('[^0-9]', '', row[2])
            completion = int(completion_numeric) if completion_numeric != '' else 0
            correct = a * b  # Calculate the correct solution by multiplying a and b
            correctness_grid[b][a] = 1 if completion == correct else -1

            # Calculate the number of digits that the completion is off by
            digits_off = abs(len(str(completion)) - len(str(correct)))
            if completion == correct:
                digits_off = 0
            elif completion != correct:
                digits_off += sum([1 for c1, c2 in zip(str(completion), str(correct)) if c1 != c2])

            digits_off_grid[b][a] = digits_off

    # Create a correctness heatmap using matplotlib
    plt.clf()
    plt.imshow(correctness_grid, cmap='plasma', extent=[0, 99, 99, 0])
    plt.colorbar(label='Correctness')
    plt.xlabel('B')
    plt.ylabel('A')
    plt.title(f'{model_name}: A * B Correctness')
    plt.savefig(f'plots/correctness {model_name}.png')
    # plt.show()

    # Create a digits-off-by heatmap using matplotlib and inverted coloring
    plt.clf()
    plt.imshow(digits_off_grid, cmap='plasma_r', extent=[0, 99, 99, 0], vmin=0, vmax=4)
    plt.colorbar(label='Digits Off By')
    plt.xlabel('B')
    plt.ylabel('A')
    plt.xticks(range(0, 100, 10))
    plt.yticks(range(0, 100, 10))
    plt.title(f'{model_name}: A * B Digits Off By')
    plt.savefig(f'plots/digits off by {model_name}.png')
    # plt.show()

    # Compute and print overall accuracy
    correct_count = 0
    incorrect_count = 0
    for row in correctness_grid:
        for value in row:
            if value == 1:
                correct_count += 1
            elif value == -1:
                incorrect_count += 1
    print(f'{model_name}: {correct_count} correct, {incorrect_count} incorrect, {correct_count / (correct_count + incorrect_count) * 100}% accuracy')


def example_plotting():
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate a 1x100 array of evenly-spaced numbers between 0 and 1
    arr = np.linspace(0, 1, 100)

    # Repeat the array to create a 100x100 grid
    grid = np.tile(arr, (100, 1))

    # Use matplotlib to create a heatmap of the grid
    plt.imshow(grid)

    # Add a colorbar to show the scale of the colors
    plt.colorbar(label='Value')

    # Add axis labels
    plt.xlabel('X')
    plt.ylabel('Y')

    # Add a title to the plot
    plt.title('100x100 grid of a pleasant gradient')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Find all files in the data folder
    filenames = []
    for filename in os.listdir('./data'):
        if filename.endswith('.csv'):
            filenames.append(filename)

    for filename in filenames:
        plot_heatmap(f'./data/{filename}')
