import torchvision
from torchvision import transforms

data = torchvision.datasets.FashionMNIST(download=False, train=True, root="../data").train_data.float()

transform_n = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((data.mean()/255,), (data.std()/255,))
])

transform_t = transforms.Compose([
    transforms.ToTensor()
])

transform_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(28),
    transforms.Normalize((data.mean()/255,), (data.std()/255,))
])