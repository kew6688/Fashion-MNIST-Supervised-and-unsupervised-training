import torch
import torchvision

class DataLoader():
    def __init__(self, transform, batch_size, num_workers):
        imagenet_data = torchvision.datasets.FashionMNIST(root='../data/', download=True)
        self.data_loader = torch.utils.data.DataLoader(transform(imagenet_data),
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=num_workers)
        return self.data_loader
