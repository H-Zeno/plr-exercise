import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """A convolutional neural network architecture for classifying MNIST digits.

    Inherits from `nn.Module`, the base class for all neural network modules in PyTorch.
    """

    def __init__(self):
        """
        Initialize the network layers.

        The network architecture is as follows:
        - First convolutional layer with 1 input channel and 32 output channels, using a 3x3 kernel and stride of 1.
        - Second convolutional layer with 32 input channels and 64 output channels, using a 3x3 kernel and stride of 1.
        - Dropout layers after convolutional layers for regularization.
        - Two fully connected layers for classification.
        """

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Defines the computation performed at every call of the network.

        Args:
            x (torch.Tensor): The input data, a tensor of shape (N, C, H, W) where
                N is the batch size,
                C is the number of channels,
                H is the height,
                W is the width.

        Returns:
            torch.Tensor: The output of the network, a tensor representing log probabilities of each class.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
