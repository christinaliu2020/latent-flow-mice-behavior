# models/reconstructor.py
import torch
import torch.nn as nn
import numpy as np
from custom_utils import gumbel_sigmoid, gumbel_softmax
import math 
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange

class ConvEncoder3_Unsuper(nn.Module):
    def __init__(self, s_dim, n_cin, n_hw, latent_size):
        super().__init__()
        self.s_dim = s_dim
        self.latent_size = latent_size #number of motions
        self.encoder =nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim, kernel_size=1, stride=1, padding=0),
        )
        self.temp_min = 0.05
        self.ANNEAL_RATE = 0.00003
        self.temp_ini = 1.0
        self.temp = 1.0
        self.linear_y = nn.Linear(s_dim, self.latent_size)
    def forward(self, input, iter):
        if iter % 100 == 1:
            self.temp = np.maximum(self.temp_ini * np.exp(-self.ANNEAL_RATE * iter), self.temp_min)
        x = self.encoder(input)
        x = x.view(-1, self.s_dim)
        z_temp1 = self.linear_y(x)
        z1 = gumbel_sigmoid(z_temp1, temperature=self.temp, categorical_dim=self.latent_size, hard=False)
        return z1


class ConvEncoder3_Unsuper_v2(nn.Module):
    """
    Reconstructor that infers a K-dimensional multi-hot spike vector y_t in (0,1)^K
    from a two-frame input [x_t, x_{t+1}] concatenated on channels.
    """
    def __init__(self, s_dim, n_cin, n_hw, latent_size,
                 tau_start=1.0, tau_min=0.05, anneal_rate=3e-5, hard=False):
        super().__init__()
        self.latent_size = latent_size
        self.hard = hard

        # A slightly deeper, normalized conv tower
        self.conv = nn.Sequential(
            nn.Conv2d(n_cin, s_dim, 4, 2, 1),      # H/2
            nn.GroupNorm(8, s_dim),
            nn.GELU(),
            nn.Conv2d(s_dim, s_dim*2, 4, 2, 1),    # H/4
            nn.GroupNorm(8, s_dim*2),
            nn.GELU(),
            nn.Conv2d(s_dim*2, s_dim*3, 4, 2, 1),  # H/8
            nn.GroupNorm(8, s_dim*3),
            nn.GELU(),
            nn.Conv2d(s_dim*3, s_dim*4, 4, 2, 1),  # H/16
            nn.GroupNorm(8, s_dim*4),
            nn.GELU(),
        )

        # Pool to 1x1 no matter the input size (more robust than using kernel=n_hw//4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → [B, s_dim*4, 1, 1]
        self.proj = nn.Sequential(
            nn.Conv2d(s_dim*4, s_dim, 1),
            nn.GELU(),
        )
        self.head = nn.Linear(s_dim, latent_size)  # logits for each of K spikes

        # Temperature schedule
        self.register_buffer("tau", torch.tensor(float(tau_start)))
        self.tau_start = float(tau_start)
        self.tau_min = float(tau_min)
        self.anneal_rate = float(anneal_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _anneal(self, step: int | None):
        if step is None:
            return float(self.tau)
        new_tau = max(self.tau_min, self.tau_start * math.exp(-self.anneal_rate * step))
        self.tau.fill_(new_tau)
        return new_tau

    def forward(self, x_pair: torch.Tensor, step: int | None = None,
                tau: float | None = None, hard: bool | None = None):
        """
        x_pair: [B, 2*C, H, W]  (concat of [x_t, x_{t+1}] on channels)
        step:   int global iteration (for annealing)
        tau:    override temperature
        hard:   override hard/soft sampling
        returns:
            y:      [B, K]  (Gumbel-Sigmoid samples in (0,1))
            logits: [B, K]
        """
        h = self.conv(x_pair)           # [B, s*4, H/16, W/16]
        h = self.pool(h)                # [B, s*4, 1, 1]
        h = self.proj(h).flatten(1)     # [B, s_dim]
        logits = self.head(h)           # [B, K]

        use_tau = float(self._anneal(step)) if tau is None else float(tau)
        use_hard = self.hard if hard is None else bool(hard)
        #y = gumbel_sigmoid(logits, temperature=use_tau, categorical_dim=self.latent_size, hard=use_hard)
        y = F.gumbel_softmax(logits, tau=use_tau, hard=use_hard, dim=-1)
        return y, logits

def conv_block(cin, cout, ks=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=ks, stride=stride, padding=padding),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvEncoder3_Unsuper_spikeslab(nn.Module):

    def __init__(self, s_dim, n_cin, n_hw, latent_size):
        super().__init__()

        self.s_dim = s_dim
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(s_dim),

            nn.Conv2d(s_dim, s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(s_dim * 2),

            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(s_dim * 2, s_dim, kernel_size=1, stride=1, padding=0),
        )
        self.temp_min = 0.1 #0.3 #0.05
        self.ANNEAL_RATE = 0.0005 #0.0005 #0.0003 #0.00003
        self.temp_ini = 1.0
        self.temp = 1.0
        #Spike infer
        self.linear_y = nn.Linear(s_dim, self.latent_size)
        #Slab infer
        self.encoder_g = nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim, kernel_size=1, stride=1, padding=0),
        )
        self.linear_g = nn.Linear(s_dim, self.latent_size)
        self.act_g = nn.Sigmoid()

    def forward(self, input, iter=None):
        if iter is not None and iter % 100 == 1:
            self.temp = np.maximum(self.temp_ini * np.exp(-self.ANNEAL_RATE * iter), self.temp_min)
        x = self.encoder(input)
        x = x.view(-1, self.s_dim)
        z_temp1 = self.linear_y(x)
        z1 = gumbel_softmax(z_temp1, temperature=self.temp, categorical_dim=self.latent_size)

        x_g = self.encoder_g(input)
        x_g = x_g.view(-1, self.s_dim)
        g = 2*self.act_g(self.linear_g(x_g))
        return z1,g

class TransformerReconstructor(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=6, dim=512, depth=6, heads=8, mlp_dim=1024, latent_size=3):
        super().__init__()
        self.latent_size = latent_size
        self.temp_ini = 1.0
        self.temp_min = 0.05
        self.ANNEAL_RATE = 0.00003
        self.temp = self.temp_ini

        # Patch embedding
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Unfold(kernel_size=patch_size, stride=patch_size),
        #     nn.Linear(patch_dim, dim)
        # )
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear_proj = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True),
            num_layers=depth
        )
        
        self.norm = nn.LayerNorm(dim)
        # Spike head
        self.linear_y = nn.Linear(dim, latent_size)
        # Slab head
        self.linear_g = nn.Linear(dim, latent_size)
        self.act_g = nn.Sigmoid()

    def forward(self, input, iter=None):
        # input: [B, C=6, H, W]
        if iter is not None and iter % 100 == 1:
            self.temp = np.maximum(self.temp_ini * np.exp(-self.ANNEAL_RATE * iter), self.temp_min)

        B, C, H, W = input.shape
        #patches = self.to_patch_embedding(input)  # [B, patch_dim, num_patches]
        patches = self.unfold(input)  # [B, patch_dim, num_patches]
        patches = patches.transpose(1, 2)  # [B, num_patches, patch_dim]
        patches = self.linear_proj(patches)  # [B, num_patches, dim]
        patches = patches + self.pos_embedding[:, :patches.size(1)]
 
        x = self.transformer(patches)  # [B, num_patches, dim]
        x = x.mean(dim=1)  # global average pooling

        ##ADDED NORMLIZATION
        x = self.norm(x)
        z_logits = self.linear_y(x)
        z1 = gumbel_softmax(z_logits, temperature=self.temp, categorical_dim=self.latent_size)

        g = 2 * self.act_g(self.linear_g(x))
        return z1, g



class CNNTransformerReconstructor(nn.Module):
    def __init__(self, img_size=256, in_channels=6, dim=512, latent_size=3, depth=4, heads=8, mlp_dim=1024):
        super().__init__()
        self.latent_size = latent_size
        self.temp_ini = 2.0
        self.temp_min = 0.7
        self.ANNEAL_RATE = 0.00001
        self.temp = self.temp_ini

        self.input_fusion = nn.Conv2d(in_channels, 3, kernel_size=1)
        
        # CNN backbone for encoding
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity() # output: [B, 512]
        self.proj = nn.Linear(512, dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

        # Spike + Slab heads
        self.linear_y = nn.Linear(dim, latent_size)
        self.linear_g = nn.Linear(dim, latent_size)
        self.act_g = nn.Sigmoid()

    def forward(self, input, iter=None):
        # input: [B, 6, H, W] (x_t, x_t+1, and/or diff)
        if iter is not None and iter % 100 == 1:
            self.temp = max(self.temp_ini * np.exp(-self.ANNEAL_RATE * iter), self.temp_min)

        B = input.shape[0]

        # CNN feature extraction
        fused_input = self.input_fusion(input)
        x = self.backbone(fused_input) # [B, 512]
        x = self.proj(x).unsqueeze(1) # [B, 1, D]

        x = self.transformer(x).squeeze(1) # [B, D]
        x = self.norm(x)

        z_logits = self.linear_y(x) # [B, latent_size]
        z1 = gumbel_softmax(z_logits, temperature=self.temp, categorical_dim=self.latent_size, hard=True)
        g = 2 * self.act_g(self.linear_g(x)) # [B, latent_size]

        return z1, g