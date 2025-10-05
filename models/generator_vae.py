# models/generator_vae.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.encoder_decoder import Encoder, Decoder, Decoder_Film
from models.encoder_decoder import EncoderGN, DecoderGN
from models.encoder_decoder import EncoderUNetGN, DecoderUNetGN
from models.encoder_decoder import EncoderResNet50, Decoder_Bkind
from models.encoder_decoder import TransformerEncoder, TransformerDecoder
from models.encoder_decoder import DinoEncoder, UNetDecoder
from models.encoder_decoder import PretrainedVJEPAEncoder
class VAE_with_kps(nn.Module):
    def __init__(self, in_channels, latent_dim, img_size,
                 cond_dim=None, enc_use_film1d=True, enc_use_film2d=True, dec_film_on_z=False):
        super().__init__()
        self.encoder = Encoder(latent_dim, in_channels, img_size,
                               cond_dim=cond_dim, use_film1d=enc_use_film1d, use_film2d=enc_use_film2d)
        self.decoder = Decoder_Film(latent_dim, in_channels, img_size,
                               cond_dim=cond_dim, film_on_z=dec_film_on_z, film_on_feats=False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond=None):
        mu, logvar = self.encoder(x, cond=cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond=cond)
        return recon, mu, logvar, z

    def inference(self, z, cond=None):
        return self.decoder(z, cond=cond)


class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, img_size):
        super().__init__()
        self.encoder = Encoder(latent_dim, in_channels, img_size)
        self.decoder = Decoder(latent_dim, in_channels, img_size)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond=None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def inference(self, z, cond=None):
        return self.decoder(z)

class VAE_GN(nn.Module):
    def __init__(self, in_channels, latent_dim, img_size):
        super().__init__()
        self.encoder = EncoderGN(latent_dim, in_channels, img_size)
        self.decoder = DecoderGN(latent_dim, in_channels, img_size)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
    def inference(self, z):
        return self.decoder(z)


class VAE_UNetGN(nn.Module):
    def __init__(self, in_channels, latent_dim, img_size):
        super().__init__()
        self.encoder = EncoderUNetGN(latent_dim, in_channels, img_size)
        self.decoder = DecoderUNetGN(latent_dim, in_channels, img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar, skips = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, skips)
        return recon, mu, logvar, z

    def inference(self, z):
        # Used only for traversals, so skip features are not used
        bs = z.size(0)
        dummy_skips = [torch.zeros(bs, c, s, s, device=z.device)
                       for c, s in zip([32, 64, 128, 256], [112, 56, 28, 14])]
        return self.decoder(z, dummy_skips)

class VAE_Bkind(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, img_channels=3):
        super().__init__()
        self.encoder = EncoderResNet50(latent_dim=latent_dim, in_channels=in_channels)
        self.decoder = Decoder_Bkind(z_dim=latent_dim, out_channels=img_channels)

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


class VAE_Transformer(nn.Module):
    def __init__(self, embed_dim=512, img_size=224, patch_size=16):
        super().__init__()
        self.encoder = TransformerEncoder(img_size=img_size, embed_dim=embed_dim, patch_size=patch_size)
        self.decoder = TransformerDecoder(embed_dim=embed_dim, img_size=img_size, patch_size=patch_size)


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


class DinoUNetVAE(nn.Module):
    def __init__(self, embed_dim, img_size=224):
        super().__init__()
        self.encoder = DinoEncoder(latent_dim=embed_dim)
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
    

class VJEPAVAE(nn.Module):
    def __init__(self, embed_dim, img_size=224):
        super().__init__()
        self.encoder = PretrainedVJEPAEncoder()
        self.decoder = UNetDecoder(z_dim=embed_dim,img_size=img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq):  # x_seq: [B, context_frames, C, H, W]
        mu, logvar = self.encoder(x_seq)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def inference(self, z):
        return self.decoder(z)
