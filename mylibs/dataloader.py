import torch
import torchvision

class DataLoader():
    def __init__(self, transform=None, batch_size=1, num_workers=1):
        train_data = torchvision.datasets.FashionMNIST(root='../data/', download=True, train=True, transform=transform)
        test_data = torchvision.datasets.FashionMNIST(root='../data/', download=True, train=False, transform=transform)
        self.train_data_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_workers)
        self.test_data_loader = torch.utils.data.DataLoader(test_data,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=num_workers)
    
    def get_data_loaders(self):
        return self.train_data_loader, self.test_data_loader
