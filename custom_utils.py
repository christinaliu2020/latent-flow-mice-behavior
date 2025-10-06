import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg16
from torchvision.transforms.functional import resize
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize_to=(224, 224)):
        super().__init__()
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features[:16]
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.vgg = backbone

        self.resize_to = resize_to
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, recon, gt):
        """Standard perceptual loss (L1 distance between features)."""
        feats_r = self.extract_features(recon)
        feats_g = self.extract_features(gt)
        return F.l1_loss(feats_r, feats_g)

    def extract_features(self, x):
        with torch.no_grad():
            x = F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            feats = self.vgg(x)
        return feats

percep_loss_fn = VGGPerceptualLoss().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("VGG on:", next(percep_loss_fn.parameters()).device)

def torch_binom(n, k):
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask

def sample_gumbel(shape, eps=1e-20, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = torch.rand(shape, device=device)
    #U = torch.rand(shape).cuda() if torch.cuda.is_available() else torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)



def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_sigmoid_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return torch.sigmoid(y / temperature)    

def gumbel_sigmoid(logits, temperature, categorical_dim, hard=False):
    y = gumbel_sigmoid_sample(logits, temperature)
    if not hard:
        return y.view(-1, categorical_dim)
    shape = y.size()
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard[y > 0.5] = 1.0
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, categorical_dim)


def gumbel_softmax(logits, temperature, categorical_dim, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, categorical_dim)


def vae_loss_fn(recon_x, x, mean, log_var):
    recon_x = torch.clamp(recon_x, 0.0, 1.0)
    BCE = F.binary_cross_entropy(recon_x.view(x.size(0), -1), 
                                   x.view(x.size(0), -1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

def vae_loss_fn_no_kl(recon_x, x, mean, log_var):
    recon_x = torch.clamp(recon_x, 0.0, 1.0)
    BCE = F.binary_cross_entropy(recon_x.view(x.size(0), -1), 
                                   x.view(x.size(0), -1), reduction='sum')
    #KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE) / x.size(0)

def vgg_vae_loss_fn(recon_x, x, mu=None, logvar=None, alpha=0.3, gamma=0.1, beta=1.0):
    device = recon_x.device
    x = x.to(device)
    recon_x = recon_x.to(device)
    recon_x = torch.clamp(recon_x, 0.0, 1.0)
    bce = F.binary_cross_entropy(recon_x.reshape(x.size(0), -1),
                                 x.reshape(x.size(0), -1), reduction='sum') / x.size(0)
    #bce = F.binary_cross_entropy(recon_x, x, reduction='sum') / (x.size(0) * x.size(1) * x.size(2) * x.size(3))
    mse = F.mse_loss(recon_x, x, reduction="mean")
    #l1 = F.mse_loss(recon_x, x, reduction='mean')
    perceptual = percep_loss_fn(recon_x, x) / x.size(0)

    if mu is not None and logvar is not None:
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return alpha * bce + gamma * perceptual + beta * kl
    else:
        return alpha * bce + gamma * perceptual

def bce(recon_x, x):
    if (recon_x < 0).any() or (recon_x > 1).any():
        print("WARNING: recon_x has elements out of [0,1] range!")
        print("Min value:", recon_x.min().item(), "Max value:", recon_x.max().item())
    if (x < 0).any() or (x > 1).any():
        print("WARNING: target x has elements out of [0,1] range!")
        print("Target min:", x.min().item(), "Target max:", x.max().item())
    recon_x = torch.clamp(recon_x, 0.0, 1.0)
    BCE = F.binary_cross_entropy(recon_x.view(x.size(0), -1), 
                                   x.view(x.size(0), -1), reduction='sum')
    return BCE / x.size(0)
