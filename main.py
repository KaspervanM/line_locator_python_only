import pickle
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from data_utils.data_generation import load_dataset, show_image, line_on_image, GrayscaleGradient

# Define Constants
DATA_PATH = "data/dataset-1000_size-28x28_bg-0.500±0.100_seed-1_line-0.400±0.050_width-0.025±0.003.npz"
NUM_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = [400, 800, 1200, 1600]
LEARNING_RATES = [0.0008, 0.0016, 0.0024, 0.0032, 0.0040, 0.0048]
SEED_VALUES = [1, 2, 3]
LOSS_FUNCTIONS = {
    'mean_squared_error': nn.MSELoss(),
    'mean_absolute_error': nn.L1Loss(),
    'huber_loss': nn.SmoothL1Loss(),
}
OPTIMIZERS = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
}
MODEL_ARCHITECTURES = {
    'model1': nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 64),
        nn.ReLU(),
        nn.Linear(64, 4),
        nn.Sigmoid()
    ),
    'model2': nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
        nn.Sigmoid()
    ),
    'model3': nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64 * 5 * 5, 64),
        nn.ReLU(),
        nn.Linear(64, 4),
        nn.Sigmoid()
    ),
    'model4': nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64 * 5 * 5, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
        nn.Sigmoid()
    ),
    # Add more model architectures as needed
}


def reset_model_to_default_init(mdl, some_seed):
    torch.manual_seed(some_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(some_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    for m in mdl.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()


def train_and_evaluate_model(x_train, y_train, x_val, y_val, model, optimizer, loss_function, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train.unsqueeze(1).float())
        loss = loss_function(outputs, y_train.float())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        y_pred = model(x_val.unsqueeze(1).float())
        mae = mean_absolute_error(y_val.cpu().numpy(), y_pred.cpu().numpy())
        model.train()
    return mae


def main():
    print("loading data")
    images, lines = load_dataset(DATA_PATH)

    images_train = images[:int(len(images) * 0.8)]
    lines_train = lines[:int(len(lines) * 0.8)]
    images_test = images[int(len(images) * 0.8):]
    lines_test = lines[int(len(images) * 0.8):]

    # Separate the images and line information
    x = np.array(images_train)
    y = np.array(lines_train)

    x_test = np.array(images_test)
    y_test = np.array(lines_test)

    # Check if a GPU is available and set it as the default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Move your dataset and targets to the GPU
    x = torch.tensor(x, device=device)
    y = torch.tensor(y, device=device)

    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)

    best_mae = np.inf
    best_model_info = None
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for epochs in NUM_EPOCHS:
        for model_name, model_architecture in MODEL_ARCHITECTURES.items():
            for loss_name, loss_function in LOSS_FUNCTIONS.items():
                for optimizer_name, optimizer_class in OPTIMIZERS.items():
                    for lr in LEARNING_RATES:
                        for seed_value in SEED_VALUES:
                            print(
                                f"Training with epochs={epochs}, model={model_name}, loss={loss_name}"
                                f", optimizer={optimizer_name}, lr={lr}, seed={seed_value}")

                            fold_mae_scores = []
                            model = None

                            kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed_value)
                            for train_index, val_index in kf.split(x):
                                x_train, x_val = x[train_index], x[val_index]
                                y_train, y_val = y[train_index], y[val_index]

                                model = model_architecture.to(DEVICE)
                                reset_model_to_default_init(model, seed_value)

                                optimizer = optimizer_class(model.parameters(), lr=lr)
                                mae = train_and_evaluate_model(x_train, y_train, x_val, y_val, model, optimizer,
                                                               loss_function, epochs)
                                fold_mae_scores.append(mae)

                            average_mae = np.mean(fold_mae_scores)
                            print(f"Average MAE: {average_mae:.4f}")

                            if average_mae < best_mae:
                                best_mae = average_mae
                                print(f"New best MAE: {best_mae:.4f}")

                                best_hyperparameters = {
                                    'model': model_name,
                                    'loss_function': loss_name,
                                    'optimizer': optimizer_name,
                                    'learning_rate': lr,
                                    'epochs': epochs,
                                    'seed': seed_value,
                                    'mae': average_mae
                                }
                                best_model_info = {
                                    'model_architecture': model,
                                    'model_state_dict': model.state_dict(),
                                    'hyperparameters': best_hyperparameters,
                                }
                                with open(f'models/model_info_{formatted_datetime}.pkl', 'wb') as file:
                                    pickle.dump(best_model_info, file)
                            print("")

    print("Best Hyperparameters:")
    print(best_model_info['hyperparameters'])
    print("Best model architecture:")
    print(best_model_info['model_architecture'])

    model = best_model_info['model_architecture'].to(device=torch.device("cpu"))
    model.load_state_dict(best_model_info['model_state_dict'])
    model.eval()

    with torch.no_grad():
        y_pred = model(x_test.unsqueeze(1).float())

    # Calculate the Mean Absolute Error (MAE) on the testing dataset
    mae = mean_absolute_error(y_test.cpu().numpy(), y_pred.cpu().numpy())
    print("Mean Absolute Error (MAE) on Testing Dataset:", mae)

    # show the first 10 images and their predictions
    for i in range(5):
        print("Prediction:", y_pred[i], "Actual:", y_test[i])
        show_image(x_test[i].detach().numpy())
        show_image(
            line_on_image(x_test[i].detach().numpy(), (y_pred[i][0], y_pred[i][1]), y_pred[i][2] - 0.5, y_pred[i][3],
                          GrayscaleGradient(1.0), GrayscaleGradient(0.0)))


if __name__ == "__main__":
    main()
