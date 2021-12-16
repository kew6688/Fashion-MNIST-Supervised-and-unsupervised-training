import torch
import numpy as np

def train(train_loader, net, loss, optimizer, USE_GPU):
    
    loss = 0
    
    with torch.no_grad():
        for i, data in enumerate(train_loader):
        
            inputs, labels = data

            if USE_GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()
                net = net.cuda()
                
            else: 
                pass
            
            optimizer.zero_grad()
            main_loss = net.forward(inputs, labels)
            main_loss.backward()
            optimizer.step()

            loss += main_loss.item()
    
    return loss