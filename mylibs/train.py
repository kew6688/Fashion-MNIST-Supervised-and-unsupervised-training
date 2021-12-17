import torch
import numpy as np
from torch._C import dtype

def train(train_loader, net, loss_function, optimizer, USE_GPU):
    
    loss = 0
    
    for i, data in enumerate(train_loader):
    
        inputs, labels = data

        if USE_GPU:
            inputs = inputs.cuda()
            labels = labels.cuda()
            net = net.cuda()
            
        else: 
            pass
        
        optimizer.zero_grad()
        outputs = net(inputs)
        main_loss = loss_function(outputs, labels)
        main_loss.backward()
        optimizer.step()

        loss += main_loss.item()
    
    return loss