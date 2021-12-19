import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from mylibs.clustering import label_data

class CustomFashionMNIST(Dataset):
    def __init__(self, train, include_labels, transform, use_unlabeled):
        super(Dataset, self).__init__()
        raw_data = torchvision.datasets.FashionMNIST(root='../data/', download=True, train=train, transform=transform)
        self.labeled_data = [(img, label) for img, label in raw_data if label in include_labels]
        self.unlabeled_data = [(img, label) for img, label in raw_data if label not in include_labels]
        if use_unlabeled:
            self.unlabeled_data = label_data(self.unlabeled_data)
            self.labeled_data.extend(self.unlabeled_data)
            self.data = self.labeled_data
        else:
            self.data = self.labeled_data
        
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx][0]
        label = torch.tensor(self.data[idx][1])
        return img, label

    def visualize(self, row = 3, column = 4):
        plt.clf()
        plt.figure(2, figsize=(12,8))
        for i in range(row * column):
            plt.subplot(row, column, i+1)
            img, label = self.__getitem__(random.randrange(self.__len__()))
            img = transforms.ToPILImage()(img)
            plt.imshow(img)
            plt.title(label.item())
            plt.axis('off')
        plt.show()

def getDataLoaders(include_labels=range(9), transform=None, train_batch_size=64, test_batch_size=1, num_workers=1, use_unlabeled=False):
    # add validation set?
    train_set = CustomFashionMNIST(train=True, include_labels=include_labels, transform=transform, use_unlabeled=use_unlabeled)
    test_set = CustomFashionMNIST(train=False, include_labels=include_labels, transform=transform, use_unlabeled=use_unlabeled)
    train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=num_workers)
    return train_dataloader, test_dataloader