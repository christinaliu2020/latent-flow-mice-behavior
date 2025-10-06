import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import wandb
import numpy as np
from models.generator_vae import VAE, VAE_with_kps, VAE_GN, VAE_UNetGN, VAE_Bkind, VAE_Transformer, DinoUNetVAE, VJEPAVAE
from models.latent_field import LatentField, LatentFieldWithKeypoints, HelmholtzLatentField, CurlFreeFlow
from models.reconstructor import ConvEncoder3_Unsuper, ConvEncoder3_Unsuper_spikeslab, TransformerReconstructor, CNNTransformerReconstructor
from custom_utils import torch_binom, vae_loss_fn, vae_loss_fn_no_kl, vgg_vae_loss_fn, bce, gumbel_sigmoid, VGGPerceptualLoss
import os 
from dataset import VideoDataset, VideoDatasetWithKeypoints
import gc
import torch.nn.functional as F

def log_latent_traversals(generator, latent_vector, z, epoch, num_vector, every_k_epochs=5):
    """
    Logs per-flow latent traversals for the first sample in batch.
    """
    if epoch % every_k_epochs != 0:
        return
    z_base = z[:1]                                # visualize one sample
    traversal_steps = [0, 1, 2, 3, 4, 5]
    for k in range(num_vector):
        recon_images = []
        for alpha in traversal_steps:
            delta_k = latent_vector[k](z_base)
            z_trav  = z_base + alpha * delta_k
            with torch.no_grad():
                recon = generator.inference(z_trav)  # [1,3,H,W]
            recon_images.append(recon[0].cpu())

        fig, axes = plt.subplots(1, len(traversal_steps), figsize=(3*len(traversal_steps), 3))
        for j, img in enumerate(recon_images):
            img = img.clamp(0, 1)
            axes[j].imshow(img.permute(1, 2, 0).numpy())
            axes[j].axis("off")
            axes[j].set_title(f"α={traversal_steps[j]}")
        plt.suptitle(f"Latent Traversal | Flow {k} | Epoch {epoch}")
        plt.tight_layout()
        wandb.log({f"Train_Latent_Traversal/Flow_{k}_Epoch_{epoch}": wandb.Image(fig)})
        plt.close(fig)


def log_t_series_panel(plot_targets, plot_preds, y_all, epoch, iteration, max_samples=6, every_k_epochs=5):
    """
    Logs a 2-row panel (GT vs Pred) over T timesteps for up to max_samples from the first batch.
    Includes hard/soft y in the title of the first column.
    """
    if epoch % every_k_epochs != 0:
        return
    if len(plot_targets) == 0:
        return

    T = len(plot_targets)
    Kshow = min(max_samples, plot_targets[0].size(0))
    for sample_idx in range(Kshow):
        fig, axes = plt.subplots(2, T, figsize=(3*T, 6))
        axes = np.atleast_2d(axes)
        for t in range(T):
            target = plot_targets[t][sample_idx].permute(1, 2, 0).numpy()
            pred   = plot_preds[t][sample_idx].permute(1, 2, 0).numpy()

            axes[0, t].imshow(np.clip(target, 0, 1)); axes[0, t].axis("off")
            axes[1, t].imshow(np.clip(pred,   0, 1)); axes[1, t].axis("off")

            # show hard/soft y on first column
            if t == 0 and t < len(y_all):
                y_vec = y_all[t][sample_idx]  # [K]
                chosen = torch.argmax(y_vec).item()
                one_hot = [1 if k == chosen else 0 for k in range(len(y_vec))]
                y_vec_str_hard = " ".join(map(str, one_hot))
                y_vec_str_soft = " ".join([f"{val:.2f}" for val in y_vec.tolist()])
                axes[0, t].set_title("GT")
                axes[1, t].set_title(f"Pred\ny=[{y_vec_str_hard}] (hard)\ny=[{y_vec_str_soft}] (soft)")

        plt.suptitle(f"Train t-series | Sample {sample_idx} | Epoch {epoch} Iter {iteration}")
        plt.tight_layout()
        wandb.log({f"Train t-series Sample {sample_idx} (E{epoch})": wandb.Image(fig)})
        plt.close(fig)

def training_function(data_loader, optimizer_spike, optimizer_full, generator, reconstructor, latent_vector,
                      num_steps, num_vector, epoch, total_epochs, print_freq=10,
                      device='cuda', stage_transition_iter=20000, global_step_tracker=None):
    generator.train()
    reconstructor.train()
    total_loss = 0.0
    iteration = global_step_tracker[0]
    final_latent_codes = []

    for i, (data, index) in enumerate(data_loader):
        iteration += 1
        global_step_tracker[0] = iteration

        optimizer = optimizer_spike if iteration < stage_transition_iter else optimizer_full
        optimizer.zero_grad()

        (seq, kp_seq, kp_mask, context_chunk) = data
        seq = seq.to(device)
        context_chunk = context_chunk.to(device)
        kp_seq = kp_seq.to(device)

        #encode first frame
        x0 = seq[:, 0]
        kp0 = kp_seq[:, 0]
        recon_x0, mu0, logvar0, z = generator(x0)
        vae_loss = vgg_vae_loss_fn(recon_x0, x0)

        #latent traversal(for visualization)
        if i == 0:
            log_latent_traversals(generator, latent_vector, z, epoch, num_vector, every_k_epochs=5)


        init_switch_prob = 0.3
        rej_prob = (1. / num_vector 
                    + 2 * init_switch_prob * (1 - init_switch_prob) * 1. / num_vector 
                    + (init_switch_prob**2) * 1. / num_vector)
        intial_prob = rej_prob
        target_prob = 0.0

        x_t1 = x0
        kp_t1 = kp0
        plot_targets, plot_preds, y_all = [], [], []
        for t in range(1, num_steps + 1):
            x_t = x_t1.clone()
            x_t1 = seq[:, t]
            kp_t1 = kp_seq[:, t]
            x_pair = torch.cat([x_t, x_t1, x_t1-x_t], dim=1)
            y, g = reconstructor(x_pair, iteration)
            y_all.append(y)
            flow_usage = y.mean(dim=0)  # average across batch
            uz = 0.0
            recon_per_flow = []   
            for idx in range(num_vector):
                delta = latent_vector[idx](z) 
                if iteration < stage_transition_iter:
                    uz += y[:, idx:idx+1] * delta
                else:
                    uz += (y[:, idx:idx+1] * g[:, idx:idx+1]) * delta

                # flow-specific reconstruction (before combining into uz)
                z_k = z + delta
                recon_k = generator.inference(z_k)
                recon_per_flow.append(recon_k)
            z = z + uz

            pred_t1 = generator.inference(z)
            vae_loss += vgg_vae_loss_fn(pred_t1, x_t1, mu0, logvar0)


            #perceptual loss for encouraging diversity
            if num_vector > 1:
                diversity_loss = 0.0
                for k in range(num_vector):
                    for j in range(k + 1, num_vector):
                        feat_i = percep_loss_fn.extract_features(recon_per_flow[k].detach())
                        feat_j = percep_loss_fn.extract_features(recon_per_flow[j].detach())
                        diversity_loss += (1 - F.cosine_similarity(
                            feat_i.flatten(1), feat_j.flatten(1)
                        ).mean())
                diversity_loss = diversity_loss / (num_vector * (num_vector - 1) / 2)
                # normalize + weight
                vae_loss += 1.0 * diversity_loss

            #use entropy loss term when hard = False
            # entropy = -(y * (y + 1e-8).log()).sum(dim=1).mean()
            # entropy_loss = -0.01 * entropy   # λ_entropy = 0.01 (tune this)
            # vae_loss += entropy_loss

            if i == 0:
                # Store targets and predictions for first batch only
                plot_targets.append(x_t1.detach().cpu())     # shape: [B, 3, H, W]
                plot_preds.append(pred_t1.detach().cpu())    # shape: [B, 3, H, W]

            target_prob = target_prob + intial_prob
            intial_prob = (intial_prob * (1-init_switch_prob) + (1 - intial_prob) * init_switch_prob) + torch_binom(
                torch.FloatTensor([3.]).to(z),
                torch.FloatTensor([intial_prob * 3]).to(z)
                ) * (init_switch_prob ** (intial_prob * 3)) * ((1-init_switch_prob) ** (3 - intial_prob * 3)) * (
                 1. / num_vector + 2 * init_switch_prob * (1 - init_switch_prob) * 
                 1. / num_vector + (init_switch_prob**2) * 1. / num_vector)

        #categorical KL
        #flow_usage = y_set.mean(dim=(0, 1))
        y_set = torch.cat(y_all, dim=0)   # [T*B, K]
        flow_usage = y_set.mean(dim=0)
        uniform_target = torch.full_like(flow_usage, 1.0 / num_vector)
        kl_flow = torch.nn.functional.kl_div((flow_usage + 1e-8).log(), uniform_target, reduction='batchmean')
        loss = vae_loss + 0.1* kl_flow

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        final_latent_codes.append(z.detach().cpu())

        if i == 0:
            log_t_series_panel(plot_targets, plot_preds, y_all, epoch, iteration, max_samples=6, every_k_epochs=5)
        if iteration % print_freq == 0:
            wandb.log({"Train Loss": loss.item(), "Iteration": iteration, "Epoch": epoch})
            print(f"Epoch [{epoch}/{total_epochs}], Iteration {iteration}: Loss = {loss.item():.4f}")
            wandb.log({
                "spike_mean": y.mean().item()
            })
        if iteration % 10 == 0:
            for j in range(num_vector):
                wandb.log({f"spike_usage/flow_{j}": y[:, j].float().mean().item()})

    avg_loss = total_loss / len(data_loader)
    wandb.log({"Epoch Average Loss": avg_loss, "Epoch": epoch})

    #final_latent_codes_tensor = torch.cat(final_latent_codes, dim=0)
    #os.makedirs("latent_codes", exist_ok=True)
    #torch.save(final_latent_codes_tensor, f"latent_codes/final_latent_codes_epoch_{epoch}.pt")
    del x0, x_t, x_t1, z, pred_t1, uz, y, g, delta
    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss

def log_val_latent_traversals(
    generator,
    latent_vector,
    z_batch,
    epoch: int,
    num_vector: int,
    alphas: torch.Tensor,
    stage_tag: str,
    num_samples_to_log: int = 6,
):
    """
    Logs per-flow latent traversal grids for a few samples from the current batch.
    - z_batch: [B, D]
    - alphas: 1D tensor of traversal coefficients (e.g., torch.linspace(0,10,9))
    """
    B = z_batch.size(0)
    idxs = torch.randperm(B)[:num_samples_to_log].tolist()

    for sample_idx in idxs:
        z0 = z_batch[sample_idx:sample_idx + 1]

        fig, axes = plt.subplots(num_vector, len(alphas), figsize=(3 * len(alphas), 3 * num_vector))
        for fidx in range(num_vector):
            delta = latent_vector[fidx](z0)
            for j, a in enumerate(alphas):
                z_mod = z0 if float(a) == 0.0 else z0 + a * delta
                img = generator.inference(z_mod)
                axes[fidx, j].imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
                axes[fidx, j].axis("off")
            axes[fidx, 0].set_ylabel(f"Flow {fidx}", fontsize=12)

        plt.suptitle(f"Val Traversal (Epoch {epoch}) [{stage_tag}] sample {sample_idx}")
        plt.tight_layout()
        wandb.log({f"Val Traversal sample {sample_idx}": wandb.Image(fig)})
        plt.close(fig)


def validation_function(
    data_loader,
    generator,
    reconstructor,
    latent_vector,
    num_steps,
    num_vector,
    epoch,
    loss_fn,
    stage_transition_iter,
    global_step_tracker,
):
    generator.eval()
    reconstructor.eval()
    total_loss = 0.0
    num_batches = 0
    device = next(generator.parameters()).device

    spike_only = global_step_tracker[0] < stage_transition_iter

    alphas = torch.linspace(0, 10, steps=9).to(device)
    num_samples_to_log = 6

    for i, (data, index) in enumerate(data_loader):
        (seq, kp_seq, kp_mask, context_chunk) = data
        seq = seq.to(device)
        context_chunk = context_chunk.to(device)
        kp_seq = kp_seq.to(device)

        x0 = seq[:, 0]
        kp0 = kp_seq[:, 0]

        recon_x0, mu, logvar, z = generator(x0)
        total_loss += loss_fn(recon_x0, x0).item()

        x_t1 = x0
        kp_t1 = kp0

        for t in range(1, num_steps + 1):
            x_t = x_t1
            x_t1 = seq[:, t]
            kp_t1 = kp_seq[:, t]

            x_pair = torch.cat([x_t, x_t1, x_t1 - x_t], dim=1)

            y, g = reconstructor(x_pair, iter=None)

            uz = 0.0
            for idx in range(num_vector):
                delta = latent_vector[idx](z)

                if spike_only:
                    uz += y[:, idx:idx+1] * delta
                else:
                    uz += (y[:, idx:idx+1] * g[:, idx:idx+1]) * delta

            z = z + uz

            # disable grads for reconstruction loss + logging
            with torch.no_grad():
                pred_t1 = generator.inference(z)

            total_loss += loss_fn(pred_t1, x_t1, mu, logvar).item()

        num_batches += 1

        if epoch % 5 == 0 and i == 0:
            stage_tag = "spike" if spike_only else "spike+slab"
            log_val_latent_traversals(
                generator=generator,
                latent_vector=latent_vector,
                z_batch=z,
                epoch=epoch,
                num_vector=num_vector,
                alphas=alphas,
                stage_tag=stage_tag,
                num_samples_to_log=6,
            )
        break  # one batch for validation

    avg_loss = total_loss / max(num_batches, 1)
    wandb.log({
        "Validation Loss": avg_loss,
        "Epoch": epoch,
        "Val Stage": "spike+slab" if not spike_only else "spike"
    })
    del x0, x_t, x_t1, z, pred_t1, uz, y, g, delta
    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss


# --- Training setup ---
total_epochs =300
batch_size = 8
wandb.init(project="latent_flow", config={"epochs": total_epochs, "batch_size": batch_size})

# Setup VAE + reconstructor
num_vector = 4
latent_dim = 256
kp_dim = 2*7
img_size = 64
kp_window = 5
cond_dim = (2*kp_window + 1) * 14
flow_dim = 128
flow_hidden_dim = 256
#generator = VAE_with_kps(in_channels=3, latent_dim=latent_dim, img_size=img_size, cond_dim=cond_dim)
# generator = VAE_GN(in_channels=3, latent_dim=latent_dim, img_size=img_size)
#generator = VAE_Bkind(in_channels=3, latent_dim=latent_dim, img_channels=3)
#generator = VAE_Transformer(embed_dim=latent_dim, img_size=img_size, patch_size=16)
generator = DinoUNetVAE(embed_dim=latent_dim, img_size=img_size)
#generator = VJEPAVAE(embed_dim=latent_dim, img_size=img_size)
#reconstructor = ConvEncoder3_Unsuper_spikeslab(s_dim=64, n_cin=3*3, n_hw=img_size, latent_size=num_vector)
#reconstructor = TransformerReconstructor(img_size=img_size, patch_size=16, in_channels=3*3, dim=latent_dim, latent_size=num_vector)
reconstructor = CNNTransformerReconstructor(img_size=img_size, in_channels=3, latent_size=num_vector)
#latent_vector = torch.nn.ModuleList([LatentField(input_dim=latent_dim, output_dim=latent_dim, hidden_dim=flow_hidden_dim) for _ in range(num_vector)])
latent_vector = torch.nn.ModuleList([LatentFieldWithKeypoints(latent_dim=latent_dim, hidden_dim=flow_hidden_dim, flow_dim=flow_dim) for _ in range(num_vector)])
#latent_vector = torch.nn.ModuleList([HelmholtzLatentField(latent_dim=latent_dim, kp_dim=cond_dim, flow_dim=flow_dim) for _ in range(num_vector)])
#latent_vector = torch.nn.ModuleList([CurlFreeFlow(latent_dim=latent_dim, hidden_dim=flow_hidden_dim) for _ in range(num_vector)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#use pretrained VAE 
checkpoint = torch.load("vae_pretrained.pt", map_location=device)
generator.load_state_dict(checkpoint)
for param in generator.encoder.parameters():
    param.requires_grad = False
generator.to(device)
reconstructor.to(device)
latent_vector.to(device)
percep_loss_fn = VGGPerceptualLoss().to(device)

# Optimizers for spike-only and full (spike+slab) stages
params_spike = list(generator.parameters()) + list(reconstructor.parameters())
params_all = params_spike + list(latent_vector.parameters())
optimizer_spike = optim.Adam(params_spike, lr=1e-4)
optimizer_full = optim.Adam(params_all, lr=1e-4)

# KL loss
tl_index = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

global_step_tracker = [0]

# Dataloaders
video_path = "mouse079.mp4"
kp_path = "mouse079_cropped_keypoints.npy"
video_dataset = VideoDatasetWithKeypoints(
    video_path=video_path,
    seq_length=30, frame_stride=5,
    keypoints_npy=kp_path,
    kp_window=kp_window, img_size=img_size
)

#video_dataset = VideoDataset(video_path, seq_length=30, frame_stride=5, img_size=img_size)
train_size = int(0.8 * len(video_dataset))
val_size = len(video_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(video_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

stage_transition_iter = 100000
# Training loop
for epoch in range(1, total_epochs + 1):
    print(f"--- Epoch {epoch}/{total_epochs} ---")
    training_loss = training_function(train_loader, optimizer_spike, optimizer_full, generator, reconstructor, latent_vector,
                                      num_steps=6, num_vector=num_vector, epoch=epoch, total_epochs=total_epochs,
                                      stage_transition_iter=stage_transition_iter, global_step_tracker=global_step_tracker)
    validation_loss = validation_function(val_loader, generator, reconstructor, latent_vector, num_steps=6,
                                      num_vector=num_vector, epoch=epoch, loss_fn=vgg_vae_loss_fn,
                                      stage_transition_iter=stage_transition_iter, global_step_tracker=global_step_tracker)
    # if epoch % 10 == 0:
    #     torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pt")
    #     torch.save(reconstructor.state_dict(), f"reconstructor_epoch_{epoch}.pt")
    #     torch.save(latent_vector.state_dict(), f"latent_vector_epoch_{epoch}.pt")
