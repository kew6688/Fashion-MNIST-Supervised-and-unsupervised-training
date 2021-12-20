import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as models
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
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(18432, 10)
        )

    def forward(self, x):
        output = self.layers(x)
        return output
    
# Using ResNet for pretrained layers, change num_classes for labelled categories training model and full trainning model
class CustomFashionResNet(nn.Module):
    def __init__(self, color_scale = 1, num_classes = 10):
        super(CustomFashionResNet, self).__init__()

        self.res18 = models.resnet18(pretrained=True, progress=False)
        encoder = list(self.res18.children())[1:8]
        input_layer = nn.Conv2d(color_scale, 64, kernel_size=7, stride=2, padding=3, bias=False)
        pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.output_layer = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        
        self.seq_modules = nn.Sequential(
            input_layer,
            *encoder,
            pooling_layer
        )
        
    def forward(self, inp):
        output = self.seq_modules(inp)
        output = self.output_layer(output.squeeze())
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