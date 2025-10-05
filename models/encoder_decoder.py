# models/encoder_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.film import FiLM1d, FiLM2d
import torchvision.models as models
from torchvision.models.vision_transformer import vit_b_16
import timm


class DinoEncoder(nn.Module):
    def __init__(self, latent_dim, freeze_mode="all", unfreeze_blocks=2):
        """
        freeze_mode: 
            "all"      → freeze all backbone params (current behavior)
            "partial"  → unfreeze last `unfreeze_blocks`
            "none"     → train whole backbone end-to-end
        unfreeze_blocks: 
            number of last transformer blocks to unfreeze if freeze_mode="partial"
        """
        super().__init__()
        # Load pretrained DINO backbone (ViT-S/16)
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)

        backbone_out_dim = self.backbone.embed_dim

        #optional model freezing 
        if freeze_mode == "all":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif freeze_mode == "partial":
            # freeze everything first
            for p in self.backbone.parameters():
                p.requires_grad = False
            # then unfreeze the last N transformer blocks
            for blk in self.backbone.blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        elif freeze_mode == "none":
            for p in self.backbone.parameters():
                p.requires_grad = True

        # projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)
        )

    def forward(self, x):  # x: [B, 3, H, W]
        # Resize to expected DINO input size
        x_enc = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # normalization
        mean = x_enc.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = x_enc.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_enc = (x_enc - mean) / std

        feats = self.backbone(x_enc)  # [B, backbone_out_dim]
        proj  = self.projection(feats)  # [B, 2*latent_dim]
        mu, logvar = proj.chunk(2, dim=1)
        return mu, logvar

    def get_param_groups(self, lr_backbone=1e-5, lr_proj=1e-4):
        """ Return parameter groups for optimizer with different lrs """
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": self.projection.parameters(), "lr": lr_proj}
        ]


        
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class Encoder(nn.Module):
    def __init__(self, latent_dim, in_channels=3, img_size=128,
                 cond_dim=None, use_film1d=False, use_film2d=False):
        super().__init__()
        self.use_film1d = use_film1d and (cond_dim is not None)
        self.use_film2d = use_film2d and (cond_dim is not None)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 128 -> 64
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),           # 64 -> 32
            nn.BatchNorm2d(64), nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1),          # 32 -> 16
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),         # 16 -> 8
            nn.BatchNorm2d(256), nn.ReLU()
        )

        if self.use_film2d:
            # Channel-wise FiLM on the final conv map [B,256,8,8]
            self.film2d = FiLM2d(cond_dim=cond_dim, num_channels=256)
        # Infer conv output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_size, img_size)
            conv_out = self.conv(dummy_input)
            flat = conv_out.view(1, -1).shape[1]
        self.fc_pre = nn.Linear(flat, 256)

        if self.use_film1d:
            self.film1d = FiLM1d(cond_dim=cond_dim, feat_dim=256)

        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x, cond=None):
        h = self.conv(x)                     # [B,256,H/16,W/16]
        if self.use_film2d:
            if cond is None:
                raise ValueError("cond (keypoints) required when use_film2d=True")
            h = self.film2d(h, cond)         # channel-wise FiLM

        h = h.view(h.size(0), -1)            # [B, flat]
        h = F.relu(self.fc_pre(h))           # [B, 256]
        
        if self.use_film1d:
            if cond is None:
                raise ValueError("cond (keypoints) required when use_film1d=True")
            h = self.film1d(h, cond)         # vector FiLM on bottleneck

        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels=3, img_size=128,
                 cond_dim=None, film_on_z=False):
        super().__init__()
        self.film_on_z = film_on_z and (cond_dim is not None)
        if self.film_on_z:
            self.film1d = FiLM1d(cond_dim=cond_dim, feat_dim=latent_dim)

        flat = (img_size // 16) ** 2 * 256
        self.fc = nn.Linear(latent_dim, flat)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16 -> 32
            nn.BatchNorm2d(64), nn.ReLU(),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32 -> 64
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # 64 -> 128
            nn.Sigmoid()
        )
        self.img_size = img_size

    def forward(self, z, cond=None):
        if self.film_on_z:
            if cond is None:
                raise ValueError("cond (keypoints) required when film_on_z=True")
            z = self.film1d(z, cond)  # FiLM on latent vector

        x = self.fc(z)
        x = x.view(-1, 256, self.img_size // 16, self.img_size // 16)
        return self.deconv(x)

class Decoder_Film(nn.Module):
    def __init__(self, latent_dim, out_channels=3, img_size=128,
                 cond_dim=None, film_on_z=False, film_on_feats=False):
        super().__init__()
        self.img_size = img_size
        self.film_on_z = film_on_z and (cond_dim is not None)
        self.film_on_feats = film_on_feats and (cond_dim is not None)

        flat = (img_size // 16) ** 2 * 256
        self.fc = nn.Linear(latent_dim, flat)

        # FiLM for latent z
        if self.film_on_z:
            self.film1d = FiLM1d(cond_dim, latent_dim)

        # FiLM for intermediate features
        if self.film_on_feats:
            self.film2d_128 = FiLM2d(cond_dim, 128)
            self.film2d_64 = FiLM2d(cond_dim, 64)
            self.film2d_32 = FiLM2d(cond_dim, 32)

        # Deconv + residual tower
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            ResidualBlock(128),
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            ResidualBlock(64),
        )

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            ResidualBlock(32),
        )

        self.deconv_final = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, cond=None):
        if self.film_on_z:
            if cond is None:
                raise ValueError("cond (keypoints) required when film_on_z=True")
            z = self.film1d(z, cond)

        x = self.fc(z)
        x = x.view(-1, 256, self.img_size // 16, self.img_size // 16)

        x = self.deconv_1(x)
        if self.film_on_feats: x = self.film2d_128(x, cond)

        x = self.deconv_2(x)
        if self.film_on_feats: x = self.film2d_64(x, cond)

        x = self.deconv_3(x)
        if self.film_on_feats: x = self.film2d_32(x, cond)

        x = self.deconv_final(x)
        return x



class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, img_channels=3, initial_size=4, initial_filters=256):
        super().__init__()
        self.init_size = initial_size
        self.fc = nn.Linear(latent_dim, initial_filters * initial_size * initial_size)

        self.decode = nn.Sequential(
            nn.BatchNorm2d(initial_filters),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(initial_filters, initial_filters // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(initial_filters // 2, initial_filters // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(initial_filters // 4, initial_filters // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(initial_filters // 8, img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), -1, self.init_size, self.init_size)
        return self.decode(x)


def conv_gn_block(cin, cout, k=4, s=2, p=1, groups=8):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p),
        nn.GroupNorm(groups, cout),
        nn.GELU()
    )

class ResidualBlockGN(nn.Module):
    def __init__(self, c, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(groups, c), nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(groups, c)
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))

class EncoderGN(nn.Module):
    def __init__(self, latent_dim, in_channels=3, img_size=224):
        super().__init__()
        C = 256
        self.conv = nn.Sequential(
            conv_gn_block(in_channels, 32),
            conv_gn_block(32, 64),
            ResidualBlockGN(64),
            conv_gn_block(64, 128),
            conv_gn_block(128, C),
            ResidualBlockGN(C),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)     # [B, C, 1, 1]
        self.fc   = nn.Sequential(
            nn.Linear(C, 512), nn.GELU(),
        )
        self.fc_mu     = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        h = self.fc(h)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

class DecoderGN(nn.Module):
    def __init__(self, latent_dim, out_channels=3, img_size=224):
        super().__init__()
        C = 256
        h = img_size // 16              # 224→14
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, C), nn.GELU(),
            nn.Linear(C, C*h*h), nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(C, 128, 4, 2, 1), nn.GroupNorm(8,128), nn.GELU(),  # 14→28
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8,64), nn.GELU(),  # 28→56
            ResidualBlockGN(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8,32), nn.GELU(),   # 56→112
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),                        # 112→224
            nn.Sigmoid(),  # keep [0,1]
        )
        self.h = h
        self.C = C

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.C, self.h, self.h)
        return self.deconv(x)


class EncoderUNetGN(nn.Module):
    def __init__(self, latent_dim, in_channels=3, img_size=224):
        super().__init__()
        self.down1 = nn.Sequential(conv_gn_block(in_channels, 32), ResidualBlockGN(32))   # 224 → 112
        self.down2 = nn.Sequential(conv_gn_block(32, 64, s=2), ResidualBlockGN(64))        # 112 → 56
        self.down3 = nn.Sequential(conv_gn_block(64, 128, s=2), ResidualBlockGN(128))      # 56 → 28
        self.down4 = nn.Sequential(conv_gn_block(128, 256, s=2), ResidualBlockGN(256))     # 28 → 14

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(256, 512), nn.GELU())
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x1 = self.down1(x)   # 112
        x2 = self.down2(x1)  # 56
        x3 = self.down3(x2)  # 28
        x4 = self.down4(x3)  # 14

        pooled = self.pool(x4).flatten(1)
        h = self.fc(pooled)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar, [x1, x2, x3, x4]

class DecoderUNetGN(nn.Module):
    def __init__(self, latent_dim, out_channels=3, img_size=224):
        super().__init__()
        self.h = img_size // 16  # 14
        self.C = 256

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.GELU(),
            nn.Linear(512, self.C * self.h * self.h), nn.GELU()
        )

        self.up4 = nn.Sequential(  # 14 → 28
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.GroupNorm(8, 128), nn.GELU()
        )
        self.up3 = nn.Sequential(  # 28 → 56
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.GroupNorm(8, 64), nn.GELU()
        )
        self.up2 = nn.Sequential(  # 56 → 112
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.GroupNorm(8, 32), nn.GELU()
        )
        self.up1 = nn.Sequential(  # 112 → 224
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, skips):
        x = self.fc(z).view(-1, self.C, self.h, self.h)  # [B, 256, 14, 14]
        x = torch.cat([x, skips[3]], dim=1)  # + down4 → [B, 512, 14, 14]
        x = self.up4(x)

        x = torch.cat([x, skips[2]], dim=1)  # + down3 → [B, 256, 28, 28]
        x = self.up3(x)

        x = torch.cat([x, skips[1]], dim=1)  # + down2 → [B, 128, 56, 56]
        x = self.up2(x)

        x = torch.cat([x, skips[0]], dim=1)  # + down1 → [B, 64, 112, 112]
        x = self.up1(x)

        return x

class EncoderResNet50(nn.Module):
    def __init__(self, latent_dim=512, in_channels=1):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        if in_channels == 1:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

    def forward(self, x):
        h = self.feature_extractor(x)  # [B, 2048, H/32, W/32]
        h = self.pool(h).view(x.size(0), -1)  # [B, 2048]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder_Bkind(nn.Module):
    def __init__(self, z_dim=2048, out_channels=3):
        super().__init__()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.initial_upsample = nn.Sequential(
            nn.Linear(z_dim, 1024 * 16 * 16),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),         # 16 → 32
            conv_block(1024, 512),
            nn.Upsample(scale_factor=2, mode='bilinear'),         # 32 → 64
            conv_block(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear'),         # 64 → 128
            conv_block(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear'),         # 128 → 256
            conv_block(128, 64),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Sigmoid()  # pixel values in [0,1]
        )

    def forward(self, z):
        x = self.initial_upsample(z)          # [B, 1024*16*16]
        x = x.view(-1, 1024, 16, 16)          # [B, 1024, 16, 16]
        return self.decoder(x)


class TransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=512, depth=6, n_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2


        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))


        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)


        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)


    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2) # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        cls = x[:, 0] # take the CLS token
        mu = self.fc_mu(cls)
        logvar = self.fc_logvar(cls)
        return mu, logvar


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=512, patch_size=16, img_size=224, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2


        self.fc = nn.Linear(embed_dim, embed_dim * self.n_patches)
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=4)


        self.to_patch = nn.Linear(embed_dim, patch_size * patch_size * out_channels)


    def forward(self, z):
        B = z.size(0)
        x = self.fc(z).view(B, self.n_patches, -1) # [B, N, D]
        x = self.transformer(x)
        x = self.to_patch(x) # [B, N, P*P*C]
        x = x.view(B, self.img_size // self.patch_size, self.img_size // self.patch_size, -1)
        x = x.permute(0, 3, 1, 2) # [B, C*P*P, H/P, W/P]
        x = x.reshape(B, -1, self.img_size, self.img_size) # [B, C, H, W]
        x = torch.sigmoid(x) 
        return x


class UNetDecoder(nn.Module):
    def __init__(self, z_dim=768, base_channels=64, out_channels=3, img_size=256):
        super().__init__()
        self.init_h = img_size // 16  # e.g., 16x16
        self.init_w = img_size // 16
        self.base_channels = base_channels
        self.fc = nn.Linear(z_dim, base_channels * 8 * self.init_h * self.init_w)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)  # [B, base_channels*8*init_h*init_w]
        x = x.view(B, self.base_channels*8, self.init_h, self.init_w)  # [B, 512, 14, 14] if base=64
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = torch.sigmoid(self.out_conv(x))
        return x

from transformers import AutoVideoProcessor, AutoModel
class PretrainedVJEPAEncoder(nn.Module):
    def __init__(self, hf_model_name="facebook/vjepa2-vitl-fpc64-256"):
        super().__init__()
        self.processor = AutoVideoProcessor.from_pretrained(hf_model_name)
        self.encoder = AutoModel.from_pretrained(hf_model_name)
        self.encoder.eval()

        embed_dim = self.encoder.config.hidden_size
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_seq):  # x_seq: [B, T, C, H, W]
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        inputs = self.processor(
            list(x_seq.cpu()),  # list of (T, C, H, W)
            return_tensors="pt"
        )["pixel_values_videos"].to(device)

        # returns [B, num_patches, D]
        outputs = self.encoder.get_vision_features(inputs)

        if outputs.dim() == 3:
            # pool across patches
            outputs = outputs.mean(dim=1)   # [B, D]

        mu = self.fc_mu(outputs)
        logvar = self.fc_logvar(outputs)
        return mu, logvar