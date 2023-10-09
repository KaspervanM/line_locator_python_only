import os
from sklearn.metrics import mean_absolute_error
from torch import optim
import torch.utils.data as data

from data_utils.data_generation import generate_data, LinearRNG, GaussianRNG, GrayscaleNumberGenerator, show_image, \
    line_on_image, GrayscaleGradient
import math
import numpy as np
import torch
import pickle
from LineRegressionModel import LineRegressionModel, custom_loss


def show_prediction(image, y_pred, y_target):
    print("Prediction:", y_pred, "Actual:", y_target, "MAE:", mean_absolute_error(y_pred, y_target))
    show_image(image)
    try:
        show_image(
            line_on_image(image, (y_pred[0], y_pred[1]), y_pred[2] - 0.5, y_pred[3],
                          GrayscaleGradient(1.0), GrayscaleGradient(0.0)))
    except ValueError:
        print("line out of range, skipping image")


def show_predictions(images, y_pred, y_target):
    for i in range(len(images)):
        show_prediction(images[i], y_pred[i], y_target[i])


def generate_dataset():
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
    x = np.array([item[0] for item in dataset_list])
    y = np.array([item[1] for item in dataset_list])

    return x, y


def get_most_recent_file():
    # Specify the directory where your files are located
    directory_path = "models"

    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter the list to include only files with a specific prefix (e.g., "output_")
    file_prefix = "cleaner_model_info_"
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

    print(f"Most recent file: {file_path}")
    return file_path


def main():
    file_path = get_most_recent_file()
    # Load the entire model_info dictionary from the pickle file
    with open(file_path, 'rb') as file:
        model_info = pickle.load(file)

    # Train the final model with the best hyperparameters on the entire dataset
    model = model_info['model'].to(device=torch.device("cpu"))

    x_test, y_target = generate_dataset()

    x_test = torch.tensor(x_test)

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
    print(x_test)
    print(y_pred)
    y_pred = y_pred.detach().numpy()

    # Calculate the Mean Absolute Error (MAE) on the testing dataset
    mae = mean_absolute_error(y_target, y_pred)
    print("Mean Absolute Error (MAE) on Testing Dataset:", mae)

    show_predictions(x_test.numpy()[:10], y_pred[:10], y_target[:10])


if __name__ == "__main__":
    main()
