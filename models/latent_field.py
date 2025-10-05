import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualMLPBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        out += residual
        return F.relu(out)

class LatentField(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=4, hidden_dim=128):
        super(LatentField, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
                
    def forward(self, x):
        #x = F.normalize(delta, dim=-1)
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


class LatentFieldWithKeypoints(nn.Module):
    def __init__(self, latent_dim=1024, kp_dim=154, flow_dim=128, num_blocks=4, hidden_dim=512):
        """
        latent_dim: dimension of z from encoder (e.g. 1024 for V-JEPA)
        kp_dim: dimension of flattened keypoints
        flow_dim: reduced working dimension for flows (smaller = easier to disentangle)
        hidden_dim: hidden size inside the flow MLP
        """
        super().__init__()
        self.down = nn.Linear(latent_dim, flow_dim)  # project down z
        #self.kp_proj = nn.Linear(kp_dim, flow_dim)   # project keypoints into same space

        # flow MLP in reduced dimension
        self.input_layer = nn.Linear(flow_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flow_dim)
        )

        self.up = nn.Linear(flow_dim, latent_dim)  # project back to full latent_dim

    def forward(self, z):
        """
        z: [B, latent_dim] (e.g., 1024)
        keypoints: [B, kp_dim] (e.g., 154 for 14*11 keypoints flattened)
        """
        z_low = self.down(z)              # [B, flow_dim]
        #kp_low = self.kp_proj(keypoints)  # [B, flow_dim]
        z_low = F.layer_norm(z_low, z_low.shape[1:])
        #kp_low = F.layer_norm(kp_low, kp_low.shape[1:])
        #x = torch.cat([z_low, kp_low], dim=-1)  # [B, 2*flow_dim]
        x = self.input_layer(z_low)
        x = self.blocks(x)
        delta_low = self.output_layer(x)        # [B, flow_dim]

        delta = self.up(delta_low)              # [B, latent_dim], matches z
        return delta

# class LatentField(nn.Module):
#     def __init__(self, latent_dim, num_heads=4, num_layers=2, dim_feedforward=512, dropout=0.1):
#         super(LatentField, self).__init__()
#         encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, 
#                                                    dim_feedforward=dim_feedforward, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc_out = nn.Linear(latent_dim, latent_dim)
    
#     def forward(self, x):
#         # x shape [batch_size, latent_dim]
#         x = x.unsqueeze(0)  # [1, batch_size, latent_dim]
#         out = self.transformer_encoder(x)  # [1, batch_size, latent_dim]
#         out = out.squeeze(0)  #[batch_size, latent_dim]
#         out = self.fc_out(out)
#         return out


class HelmholtzLatentField(nn.Module):
    def __init__(self, latent_dim=1024, kp_dim=154, flow_dim=256, hidden_dim=512, num_blocks=4):
        """
        Helmholtz decomposition for latent flows:
        - Gradient part (∇Φ): curl-free
        - Divergence-free part (r(z)): divergence-free
        """
        super().__init__()
        self.down = nn.Linear(latent_dim, flow_dim)   # project z down
        self.kp_proj = nn.Linear(kp_dim, flow_dim)    # project keypoints

        # Shared base
        self.input_layer = nn.Linear(2*flow_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])

        # Helmholtz heads
        self.grad_head = nn.Linear(hidden_dim, flow_dim)  # gradient part
        self.div_head = nn.Linear(hidden_dim, flow_dim)   #divergence-free part

        # Project back up to latent_dim
        self.up = nn.Linear(flow_dim, latent_dim)

    def forward(self, z, keypoints):
        """
        z: [B, latent_dim]
        keypoints: [B, kp_dim]
        Returns:
            delta: combined latent update [B, latent_dim]
            grad_flow: gradient part in latent_dim
            div_flow: divergence-free part in latent_dim
        """
        # Downproject
        z_low = self.down(z)
        kp_low = self.kp_proj(keypoints)
        z_low = F.layer_norm(z_low, z_low.shape[1:])
        kp_low = F.layer_norm(kp_low, kp_low.shape[1:])

        # Shared features
        x = torch.cat([z_low, kp_low], dim=-1)
        h = self.input_layer(x)
        h = self.blocks(h)

        # Helmholtz parts
        grad_flow_low = self.grad_head(h)   # curl-free part
        div_flow_low = self.div_head(h)     # divergence-free part

        # Combine and project back up
        delta_low = grad_flow_low + div_flow_low
        delta = self.up(delta_low)

        return delta, self.up(grad_flow_low), self.up(div_flow_low)

    def divergence_penalty(self, div_flow, z):
        """
        Approximate divergence penalty for r(z):
        L_DIV = (∇ · r(z))^2
        div_flow: [B, latent_dim] divergence-free part
        z: [B, latent_dim] original latent
        """
        # Numerical Jacobian approximation
        div_loss = torch.autograd.grad(
            outputs=div_flow.sum(), inputs=z,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]  # ∇·r(z)
        return (div_loss**2).mean()


class CurlFreeFlow(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, num_blocks=3):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(num_blocks):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))  # scalar potential u(z)
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        """
        z: [B, latent_dim]
        Returns:
            flow: [B, latent_dim] (curl-free vector field = grad u(z))
            potential: [B, 1]
        """
        
        z = z.clone().requires_grad_(True)
        u = self.mlp(z)   # [B,1]
        # gradient wrt z
        grad_u = grad(u.sum(), z, create_graph=True)[0]  # [B, latent_dim]
        return grad_u, u