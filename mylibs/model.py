import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from pl_bolts.models.autoencoders import VAE

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

        self.e = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 4)
        )

        self.d = nn.Sequential(
            nn.Linear(2, 1152),
            nn.ReLU(),
            nn.Unflatten(1, (32, 6, 6)),
            nn.ConvTranspose2d(32, 64, 3, 2, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 1, 1)
        )
        
        # self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        # self.conv2 = nn.Conv2d(32, 32, 3, padding='same')
        # self.conv3 = nn.Conv2d(32, 1, 3, padding='same')
        # self.batch_norm = nn.BatchNorm2d(32)
        # self.relu = nn.ReLU()
        # self.flatten = nn.Flatten()
        # self.unflatten = nn.Unflatten(1, (32, 28, 28))
        # self.linear1 = nn.Linear(25088, 4)
        # self.linear2 = nn.Linear(2, 25088)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder(self, x):
        # x = self.conv1(x)
        # x = self.batch_norm(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.batch_norm(x)
        # x = self.relu(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        x = self.e(x)
        return x

    def decoder(self, x):
        # x = self.linear2(x)
        # x = self.unflatten(x)
        # x = F.conv_transpose2d(x, self.conv2.weight.transpose(0, 1), padding=(1, 1))
        # x = self.batch_norm(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # x = self.relu(x)
        x = self.d(x)
        x = F.pad(x, [1, 0, 1, 0])
        return x

    def reparameterize(self, mean, logvar):
        eps = torch.normal(torch.zeros(mean.size()), torch.ones(mean.size()))
        return eps * torch.exp(logvar * 0.5) + mean


    def encode(self, images, USE_GPU):
        res = []
        for i, image in enumerate(images):

            if USE_GPU:
                image = image.cuda()
            x = self.encoder(image)
            mean, logvar = x[:, :2], x[:, 2:]
            x = self.reparameterize(mean, logvar)
            res.append(x.detach().cpu().numpy()[0])
        return np.array(res)