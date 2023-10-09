import torch.nn as nn


class LineRegressionModel(nn.Module):
    def __init__(self, num_conv_layers, num_conv_units, num_fc_layers, num_fc_units):
        super(LineRegressionModel, self).__init__()

        self.conv_layers = nn.Sequential()
        for i in range(num_conv_layers):
            in_channels = 1 if i == 0 else num_conv_units
            self.conv_layers.add_module(
                f'conv{i + 1}',
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_conv_units,
                    kernel_size=3,
                    padding=1
                )
            )
            self.conv_layers.add_module(f'relu{i + 1}', nn.ReLU())
            self.conv_layers.add_module(f'pool{i + 1}', nn.MaxPool2d(kernel_size=2, stride=2))

        # Flatten the output from convolutional layers
        self.fc_layers = nn.Sequential(nn.Flatten())
        for i in range(num_fc_layers):
            in_features = num_conv_units * (28 // (2 ** num_conv_layers)) ** 2 if i == 0 else num_fc_units
            out_features = 4 if i == num_fc_layers - 1 else num_fc_units
            self.fc_layers.add_module(f'fc{i + 1}', nn.Linear(in_features=in_features, out_features=out_features))
            if i == num_fc_layers - 1:
                self.fc_layers.add_module('sigmoid', nn.Sigmoid())
            else:
                self.fc_layers.add_module(f'relu{i + 1}', nn.ReLU())

    def forward(self, x):
        x = self.conv_layers(x.unsqueeze(1).float())
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Define a custom loss function (e.g., Mean Squared Error)
def custom_loss(outputs, targets):
    loss = nn.L1Loss()(outputs, targets)
    return loss
