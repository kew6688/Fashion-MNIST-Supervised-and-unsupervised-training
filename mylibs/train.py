import torch
import numpy as np
from torch._C import dtype
from sklearn.metrics import f1_score

def train(train_loader, net, loss_function, optimizer, USE_GPU):
    
    loss = 0
    correct = 0
    preds = []
    gt_labels = []
    
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
        # --- eval metrics ---
        pred_label = outputs.argmax(dim=1)
        correct += (pred_label == labels).sum().item()
        preds.extend(pred_label)
        gt_labels.extend(labels)
        # --- eval metrics ---
        main_loss = loss_function(outputs, labels)
        main_loss.backward()
        optimizer.step()

        loss += main_loss.item()

    gt_labels = torch.tensor(gt_labels)
    preds = torch.tensor(preds)

    eval_metrics = {"acc": correct / len(train_loader.dataset),
                    "loss": loss / len(train_loader),
                    "f1": f1_score(gt_labels, preds, average='weighted'),
                    }
    
    return eval_metrics


def autoencoder_train(train_loader, net, loss_function, optimizer, USE_GPU):
    
    loss = 0
    
    for i, inputs in enumerate(train_loader):

        if USE_GPU:
            inputs = inputs.cuda()
            net = net.cuda()
            
        else: 
            pass
        
        optimizer.zero_grad()
        outputs = net(inputs)
        # --- eval metrics ---
        main_loss = loss_function(outputs, inputs)
        main_loss.backward()
        optimizer.step()

        loss += main_loss.item()

    eval_metrics = {"loss": loss / len(train_loader.dataset)}
    
    return eval_metrics