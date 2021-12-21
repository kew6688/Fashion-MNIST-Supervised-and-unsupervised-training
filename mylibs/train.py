import torch
from torch._C import dtype
from sklearn.metrics import f1_score
from mylibs.eval import validate

def train(train_val_loaders, net, loss_function, optimizer, USE_GPU, checkpoint_path):
    
    loss = 0
    correct = 0
    preds = []
    gt_labels = []

    train_loader, val_loader = train_val_loaders
    
    for i, data in enumerate(train_loader):

        net.train()
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

    # validate
    val_metrics = validate(val_loader, net, loss_function, USE_GPU)
    train_metrics = {"acc": correct / len(train_loader.dataset),
                     "loss": loss / len(train_loader),
                     "f1": f1_score(torch.tensor(gt_labels), torch.tensor(preds), average='weighted'),
                    }
    metrics = {"train": train_metrics, 
               "val": val_metrics,
               }
    # save checkpoint for future reference
    torch.save({'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
                }, 
                checkpoint_path)
    return metrics


def autoencoder_train(train_loader, net, loss_function, optimizer, USE_GPU):
    
    loss = 0
    
    for i, inputs in enumerate(train_loader):

        if USE_GPU:
            inputs = inputs[0].cuda()
            net = net.cuda()
            
        else: 
            inputs = inputs[0]
        
        optimizer.zero_grad()
        # --- eval metrics ---
        l1_penalty = sum([p.abs().sum() for p in net.parameters()])
        l2_penalty = sum([(p**2).sum() for p in net.parameters()])
        main_loss = loss_function(inputs, net, USE_GPU)
        regularized_loss = main_loss + 0.01 * l1_penalty + 0.1 * l2_penalty
        regularized_loss.backward()
        optimizer.step()

        loss += main_loss.item()

    eval_metrics = {"loss": loss / len(train_loader)}
    
    return eval_metrics