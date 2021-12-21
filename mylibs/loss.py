import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
loss_function = nn.CrossEntropyLoss()

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = torch.log(torch.tensor(2.0 * torch.as_tensor(np.pi)))
    return torch.sum(-0.5 * ((sample - mean) ** 2.0 * torch.exp(torch.tensor(-logvar)) + logvar + log2pi), dim=raxis)

def compute_loss(model, x, USE_GPU):
    e = model.encoder(x)
    mean, logvar = e[:, :2], e[:, 2:]
    z = model.reparameterize(mean, logvar, USE_GPU)
    x_logit = model.decoder(z)
    cross_ent = F.binary_cross_entropy_with_logits(x_logit, x, reduction='none')
    logpx_z = -torch.sum(cross_ent, dim=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -torch.mean(logpx_z + logpz - logqz_x)

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
 
    def forward(self, inputs, model, USE_GPU):
        return compute_loss(model, inputs, USE_GPU)

autoencoder_loss = VAELoss()