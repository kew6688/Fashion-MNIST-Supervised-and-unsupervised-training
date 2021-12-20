import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from mylibs.clustering import label_data

class CustomFashionMNIST(Dataset):
    # mode:
    # 0 - use only labeled data
    # 1 - use labeled data and clustering-labeled unlabeled data
    # 2 - use full FasionMNIST data
    def __init__(self, train, include_labels, transform, mode=0):
        super(Dataset, self).__init__()
        raw_data = torchvision.datasets.FashionMNIST(root='../data/', download=True, train=train, transform=transform)
        self.labeled_data = [(img, label) for img, label in raw_data if label in include_labels]
        self.unlabeled_data = [(img, label) for img, label in raw_data if label not in include_labels]
        if mode == 0:
            self.data = self.labeled_data
        elif mode == 1:
            self.unlabeled_data = label_data(self.unlabeled_data, num_classes=10-len(include_labels))
            self.labeled_data.extend(self.unlabeled_data)
            self.data = self.labeled_data
        else:
            self.labeled_data.extend(self.unlabeled_data)
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

def getDataLoaders(include_labels=range(9), transform=None, batch_size=64, num_workers=1, mode=2):
    # add validation set?
    train_set = CustomFashionMNIST(train=True, include_labels=include_labels, transform=transform, mode=mode)
    test_set = CustomFashionMNIST(train=False, include_labels=include_labels, transform=transform, mode=2)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_dataloader, test_dataloader