import torch
import numpy as np
from sklearn.metrics import f1_score

def validate(val_loader, net, loss, USE_GPU):
    
    val_loss = 0
    correct = 0
    preds = []
    gt_labels = []
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
        
            inputs, labels = data

            if USE_GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()
                net = net.cuda()
                
            else: 
                pass
            
            outputs = net(inputs)
            val_loss += loss(outputs, labels)
            # --- eval metrics ---
            pred_label = outputs.argmax(dim=1)
            correct += (pred_label == labels).sum().item()
            preds.extend(pred_label)
            gt_labels.extend(labels)
            # --- eval metrics ---

    preds = torch.tensor(preds)
    gt_labels = torch.tensor(gt_labels)
    eval_metrics = {"acc": correct / len(val_loader.dataset), 
                    "loss": val_loss / len(val_loader),
                    "f1": f1_score(gt_labels, preds, average='weighted')}
    
    return eval_metrics