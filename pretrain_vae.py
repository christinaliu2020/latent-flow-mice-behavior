import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
#from models.generatr_vae import DinoUNetVAE   
from models.encoder_decoder import DinoEncoder, UNetDecoder
# from custom_utils import vgg_vae_loss_fn 
from dataset import VideoDataset, VideoDatasetWithKeypoints
import wandb
from torchvision.models import vgg16, VGG16_Weights

class DinoUNetVAE(nn.Module):
    def __init__(self, embed_dim, img_size=224):
        super().__init__()
        self.encoder = DinoEncoder(latent_dim=embed_dim, freeze_mode="partial")
        #self.encoder = EncoderResNet50(latent_dim=embed_dim, in_channels=3)
        self.decoder = UNetDecoder(z_dim=embed_dim,img_size=img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def inference(self, z):
        return self.decoder(z)

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

def vgg_vae_loss_fn(recon_x, x, mu=None, logvar=None, alpha=10, gamma=0.1, beta=1.0):
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
        total_loss = alpha * bce + gamma * perceptual + beta * kl
        return total_loss, bce, perceptual, kl
    else:
        total_loss = alpha * bce + gamma * perceptual
        return total_loss, bce, perceptual, None

def pretrain_vae(video_path, save_path="vae_pretrained.pt",
                 img_size=64, latent_dim=256, batch_size=32, epochs=50, lr=1e-3):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoDatasetWithKeypoints(video_path=video_path,
    seq_length=15, frame_stride=3,img_size=img_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = DinoUNetVAE(embed_dim=latent_dim, img_size=img_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        total_loss = 0
        for (data,idx) in dataloader:
            (seq, kp_seq, kp_mask, context_chunk) = data
            frames = seq.view(-1, seq.size(2), seq.size(3), seq.size(4))  # [B*T, 3, H, W]
            frames = frames.to(device)

            recon, mu, logvar, _ = vae(frames)
            loss, recon_loss, perceptual, kl_loss = vgg_vae_loss_fn(recon, frames, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")
        wandb.log({
            "train/total_loss": loss.item(),
            "train/recon_loss": recon_loss.item(),
            "train/kl_loss": kl_loss.item(),
            "train/perceptual_loss": perceptual.item()
        })
        # visualize some reconstructions
        if epoch % 5 == 0:
            with torch.no_grad():
                recon, _, _, _ = vae(frames[:8])
                fig, axes = plt.subplots(2, 8, figsize=(16, 4))
                for i in range(8):
                    axes[0, i].imshow(frames[i].permute(1, 2, 0).cpu().numpy())
                    axes[0, i].axis("off")
                    axes[1, i].imshow(recon[i].permute(1, 2, 0).cpu().numpy())
                    axes[1, i].axis("off")
                wandb.log({f"reconstructions/epoch_{epoch}": wandb.Image(fig)})
                plt.close(fig)

    torch.save(vae.state_dict(), save_path)
    print(f"Saved pretrained VAE to {save_path}")


if __name__ == "__main__":
    wandb.init(project="pretrain_latent_flow_vae")
    video_path = "mouse079.mp4" 
    pretrain_vae(video_path, save_path="vae_pretrained.pt",
                 img_size=64, latent_dim=256, batch_size=8, epochs=50, lr=1e-4)