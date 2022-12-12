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
