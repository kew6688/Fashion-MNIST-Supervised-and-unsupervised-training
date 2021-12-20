import torch
import torchvision.models as models

def save(nn):
    torch.save(nn, '../data/model.pth')
    
def load(nn):
    return torch.load('../data/model.pth')
    