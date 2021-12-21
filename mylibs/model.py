import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as models
import numpy as np
from pl_bolts.models.autoencoders import VAE

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
        encoder = list(self.res18.children())[1:9]
        input_layer = nn.Conv2d(color_scale, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.dropblock = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=1),
            nn.Flatten(),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256, out_features=num_classes, bias=False)
        )
        
        self.seq_modules = nn.Sequential(
            input_layer,
            *encoder,
            self.dropblock
        )
        
    def forward(self, inp):
        output = self.seq_modules(inp)
        return output

class Autoencoder(VAE):
    def __init__(self, input_height):
        super(Autoencoder, self).__init__(input_height)
        self.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder.conv1 = torch.nn.Conv2d(64*self.decoder.expansion, 1, kernel_size=3, stride=1, padding=3, bias=False)

    def encode(self, images, USE_GPU):
        res = []
        for i, image in enumerate(images):

            if USE_GPU:
                image = image.cuda()
            e = self.encoder(image).detach().cpu().numpy()[0]
            res.append(e)
        return np.array(res)