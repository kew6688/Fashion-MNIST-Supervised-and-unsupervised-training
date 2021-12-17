import torch
import torch.nn as nn
import torch.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(18432, 10)
        )

    def forward(self, x):
        x = self.layers(x)
        output = x
        return output