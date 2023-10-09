import pickle
from datetime import datetime
import torch
import torch.optim as optim
import numpy as np
import ray
from ray import tune
import torch.utils.data as data
from sklearn.model_selection import train_test_split, KFold

import test_cleaner
from LineRegressionModel import LineRegressionModel, custom_loss
from data_utils.data_generation import load_dataset

DATA_PATH = "data/dataset-1000_size-28x28_bg-0.500±0.050_seed-1_line-0.300±0.050_width-0.025±0.003.npz"
NUM_SAMPLES = 100  # Number of hyperparameter configurations to try


# Function to determine available resources on the current device
def get_resources_for_device():
    available_resources = ray.available_resources()
    resources = {"cpu": available_resources["CPU"], "gpu": available_resources["GPU"]}
    print("Resources:", resources)
    return resources


# Define a function to train and evaluate the model with k-fold cross-validation
def train_eval(config):
    # Retrieve the random seed from the hyperparameters
    # seed = config["seed"]

    # Split the combined set into training and validation using k-fold cross-validation
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)  # seed)

    # Initialize a list to store validation losses
    val_losses = []

    for train_index, val_index in kf.split(images_combined):
        # random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # np.random.seed(seed)
        train_dataset = data.TensorDataset(torch.tensor(images_combined[train_index], dtype=torch.float32),
                                           torch.tensor(lines_combined[train_index], dtype=torch.float32))
        val_dataset = data.TensorDataset(torch.tensor(images_combined[val_index], dtype=torch.float32),
                                         torch.tensor(lines_combined[val_index], dtype=torch.float32))

        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = LineRegressionModel(config["num_conv_layers"], config["num_conv_units"], config["num_fc_layers"],
                                    config["num_fc_units"])
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(config["epochs"]):
            # Training loop
            model.train()
            optimizer.zero_grad()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = custom_loss(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    val_outputs = model(inputs)
                    val_loss += custom_loss(val_outputs, targets).item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

    # Calculate and return the average validation loss across folds
    avg_val_loss = sum(val_losses) / len(val_losses)
    tune.report(loss=avg_val_loss)


if __name__ == "__main__":
    print("loading data")
    images, lines = load_dataset(DATA_PATH)
    ray.init()

    # Split your dataset into test set and combined (train + validation) set
    images_combined, images_test, lines_combined, lines_test = train_test_split(images, lines, test_size=0.2,
                                                                                random_state=42)

    # Define data loaders for training and validation combined set
    batch_size = 64  # You can adjust this as needed

    # Define the hyperparameter search space, including the random seed
    config_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": tune.choice([16, 32, 64, 128]),
        "num_conv_layers": tune.choice([1, 2, 3]),
        "num_conv_units": tune.choice([16, 32, 64]),
        "num_fc_layers": tune.choice([1, 2, 3]),
        "num_fc_units": tune.choice([64, 128, 256, 512]),
        # "seed": tune.choice([42, 123, 567, 789]),  # Different seed values to explore
    }

    # Perform hyperparameter tuning using Ray Tune
    analysis = tune.run(
        train_eval,
        num_samples=NUM_SAMPLES,
        config=config_space,
        resources_per_trial=get_resources_for_device(),
        #log_to_file=False,
        #local_dir="./ray_results"
    )

    # Get the best hyperparameters
    best_config = analysis.get_best_config(metric="loss", mode="min")

    print("Best hyperparameters:", best_config)

    # Train the final model with the best hyperparameters on the entire dataset
    final_model = LineRegressionModel(best_config["num_conv_layers"], best_config["num_conv_units"],
                                      best_config["num_fc_layers"],
                                      best_config["num_fc_units"])
    optimizer = optim.Adam(final_model.parameters(), lr=best_config["lr"])

    combined_dataset = data.TensorDataset(torch.tensor(images_combined, dtype=torch.float32),
                                          torch.tensor(lines_combined, dtype=torch.float32))
    test_dataset = data.TensorDataset(torch.tensor(images_test, dtype=torch.float32),
                                      torch.tensor(lines_test, dtype=torch.float32))

    combined_loader = data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(best_config["epochs"]):
        # Training loop
        final_model.train()
        optimizer.zero_grad()
        for inputs, targets in combined_loader:
            outputs = final_model(inputs)
            loss = custom_loss(outputs, targets)
            loss.backward()
            optimizer.step()

    # Save the trained model
    print("Saving model...")
    best_model_info = {
        'model': final_model,
        'model_state_dict': final_model.state_dict(),
        'hyperparameters': best_config,
    }

    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'models/model_info_{formatted_datetime}.pkl', 'wb') as file:
        pickle.dump(best_model_info, file)

    # Evaluate the final model on the test set
    final_model.eval()
    test_loss = 0.0
    inputs_list = []
    targets_list = []
    predictions_list = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            test_outputs = final_model(inputs)
            test_loss += custom_loss(test_outputs, targets).item()
            for inp in inputs.cpu().detach().numpy():
                inputs_list.append(np.array(inp))
            for target in targets.cpu().detach().numpy():
                targets_list.append(np.array(target))
            for pred in test_outputs.cpu().detach().numpy():
                predictions_list.append(np.array(pred))

    test_loss /= len(test_loader)
    print("Final Test Loss:", test_loss)

    test_cleaner.show_predictions(inputs_list[:10], predictions_list[:10], targets_list[:10])
