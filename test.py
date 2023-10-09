import os

from sklearn.metrics import mean_absolute_error

from data_utils.data_generation import generate_data, LinearRNG, GaussianRNG, GrayscaleNumberGenerator, show_image, \
    line_on_image, GrayscaleGradient
import math
import numpy as np
import torch
import pickle

# Specify the directory where your files are located
directory_path = "models"

# List all files in the directory
files = os.listdir(directory_path)

# Filter the list to include only files with a specific prefix (e.g., "output_")
file_prefix = "model_info_"
filtered_files = [file for file in files if file.startswith(file_prefix)]

# Sort the filtered files by their filenames (assumes filenames include a date and time)
sorted_files = sorted(filtered_files, reverse=True)

if not sorted_files:
    raise FileNotFoundError(f"No files found with the prefix '{file_prefix}' in the directory.")
# Check if any files match the prefix
# The fi\rst file in the sorted list will be the most recent
most_recent_file = sorted_files[0]

# Now you can load or process this file as needed
file_path = os.path.join(directory_path, most_recent_file)

print(f"Most recent file: {most_recent_file}")

# Load the entire model_info dictionary from the pickle file
with open(file_path, 'rb') as file:
    model_info = pickle.load(file)

# Access the model architecture
model = model_info['model_architecture'].to(device=torch.device("cpu"))

# Load the model state dictionary
model.load_state_dict(model_info['model_state_dict'])

# Access the hyperparameters
hyperparameters = model_info['hyperparameters']

# test the model on 10 newly generated images


dataset_size = 10
image_size = (28, 28)
image_nr_background_colors = LinearRNG(1, 5)
image_background_grays = GrayscaleNumberGenerator(GaussianRNG(0.5, 0.1))
seed = 3
diagonal = math.hypot(*image_size)
line_min_length = 4 / diagonal
line_nr_colors = LinearRNG(1, 5)
line_grays = GrayscaleNumberGenerator(GaussianRNG(0.4, 0.05))
line_nr_widths = LinearRNG(1, 5)
line_widths = GaussianRNG(1 / diagonal, 0.5 / diagonal, limit_left=0, limit_right=4 / diagonal)
prescale_image_size = (LinearRNG(image_size[0], image_size[0] * 30), LinearRNG(image_size[1], image_size[1] * 30))
dataset_list = generate_data(dataset_size, image_size, image_nr_background_colors, image_background_grays, seed,
                             line_min_length, line_nr_colors, line_grays, line_nr_widths, line_widths,
                             prescale_image_size=prescale_image_size)

# Separate the images and line information
X_test = np.array([item[0] for item in dataset_list])
y_test = np.array([item[1] for item in dataset_list])

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test.unsqueeze(1).float())

# Calculate the Mean Absolute Error (MAE) on the testing dataset
mae = mean_absolute_error(y_test.cpu().numpy(), y_pred.cpu().numpy())
print("Mean Absolute Error (MAE) on Testing Dataset:", mae)

# show the first 10 images and their predictions
for i in range(10):
    print("Prediction:", y_pred[i], "Actual:", y_test[i], "MAE:", mean_absolute_error(y_test[i], y_pred[i]))
    show_image(X_test[i].detach().numpy())
    try:
        show_image(
            line_on_image(X_test[i].detach().numpy(), (y_pred[i][0], y_pred[i][1]), y_pred[i][2] - 0.5, y_pred[i][3],
                          GrayscaleGradient(1.0), GrayscaleGradient(0.0)))
    except ValueError:
        print("line out of range, skipping image")
