import torch
import numpy as np

def validate(val_loader, net, loss, USE_GPU):
    
    val_loss = 0
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
        
            inputs, labels = data

            if USE_GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()
                net = net.cuda()
                
            else: 
                pass
            
            preds = net.forward(inputs)
            val_loss += loss(preds, labels)
    
    return val_loss