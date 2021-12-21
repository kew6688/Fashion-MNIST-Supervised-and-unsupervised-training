import torchvision
from torchvision import transforms

data = torchvision.datasets.FashionMNIST(download=False, train=True, root="../data").train_data.float()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((data.mean()/255,), (data.std()/255,))
])