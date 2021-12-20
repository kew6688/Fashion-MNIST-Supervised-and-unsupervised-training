import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

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

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 4, 2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(4, 1, 2, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(49, 2),
            nn.ReLU()
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 49),
            nn.ReLU(),
            nn.Unflatten(1, (1, 7, 7)),
            nn.Conv2d(1, 4, 2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(4, 16, 2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 32, 2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, images):
        return np.array([self.encoder(x).detach().numpy()[0] for x in images])