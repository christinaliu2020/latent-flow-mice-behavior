#uses gumbel_sigmoid in reconstructor for multiple flows to be active 
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import wandb
import numpy as np
from models.vae import VAE, VAE_with_kps, VAE_GN, VAE_UNetGN, VAE_Bkind, VAE_Transformer, DinoUNetVAE, VJEPAVAE
from models.latent_field import LatentField, LatentFieldWithKeypoints, HelmholtzLatentField, CurlFreeFlow
from models.conv_encoder_unsuper import ConvEncoder3_Unsuper, ConvEncoder3_Unsuper_spikeslab, TransformerReconstructor, CNNTransformerReconstructor
from custom_utils import torch_binom, vae_loss_fn, vae_loss_fn_no_kl, vgg_vae_loss_fn, bce, gumbel_sigmoid, VGGPerceptualLoss
import os
from dataset import VideoDataset, VideoDatasetWithKeypoints
import gc
import torch.nn.functional as F
import wandb

@torch.no_grad()
def log_latent_traversals(
    generator,
    reconstructor,
    latent_vector,
    seq,            # [B, T, 3, H, W]
    kp_seq,         # [B, T, K_kp]
    epoch,
    num_vector,
    num_steps,
    sample_idx=0,
    every_k_epochs=5,
    traversal_alphas=(0, 1, 2, 3, 4, 5),
):
    if epoch % every_k_epochs != 0:
        return

    device = next(generator.parameters()).device
    B, T = seq.shape[:2]
    T_use = min(num_steps, T - 1)

    # -------- Figure 1: per-flow traversals from μ(x0) --------
    x0  = seq[sample_idx:sample_idx+1, 0].to(device)
    kp0 = kp_seq[sample_idx:sample_idx+1, 0].to(device)
    _, mu0, _, _ = generator(x0)  # μ(x0)

    traversal_steps = list(traversal_alphas)
    for k in range(num_vector):
        recon_images = []
        for alpha in traversal_steps:
            delta_k = latent_vector[k](mu0, kp0)       # [1,D]
            z_trav  = mu0 + alpha * delta_k
            recon   = generator.inference(z_trav)      # [1,3,H,W]
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

    # -------- Figure 2: GT vs Pred rollout + (multi-hot) gate labels --------
    gt_frames, pred_frames, chosen_ids, y_probs_list = [], [], [], []

    x_t  = seq[sample_idx:sample_idx+1, 0].to(device)
    _, mu_t, _, _ = generator(x_t)

    for t in range(1, T_use + 1):
        x_t1  = seq[sample_idx:sample_idx+1, t].to(device)
        kp_t1 = kp_seq[sample_idx:sample_idx+1, t].to(device)

        x_pair = torch.cat([x_t, x_t1, x_t1 - x_t], dim=1)
        y, _   = reconstructor(x_pair, iter=None)  

        y_vec = y[0].detach().cpu()
        chosen_k = int(torch.argmax(y_vec).item())
        chosen_ids.append(chosen_k)
        y_probs_list.append(y_vec.tolist())

        deltas = torch.stack([latent_vector[k](mu_t, kp_t1) for k in range(num_vector)], dim=1)  # [1,K,D]
        selected_delta = (y.unsqueeze(-1) * deltas).sum(dim=1)                                    # [1,D]
        z_next_pred    = mu_t + selected_delta
        pred_t1        = generator.inference(z_next_pred)                                         # [1,3,H,W]

        gt_frames.append(x_t1[0].detach().cpu())
        pred_frames.append(pred_t1[0].detach().cpu())

        _, mu_t, _, _ = generator(x_t1)
        x_t = x_t1

    fig, axes = plt.subplots(2, T_use, figsize=(3*T_use, 6))
    if T_use == 1:
        axes = np.atleast_2d(axes)
    for j in range(T_use):
        gt  = gt_frames[j].clamp(0,1).permute(1,2,0).numpy()
        prd = pred_frames[j].clamp(0,1).permute(1,2,0).numpy()
        axes[0, j].imshow(gt);  axes[0, j].axis("off")
        axes[1, j].imshow(prd); axes[1, j].axis("off")

        if j == 0:
            axes[0, j].set_title("GT")
        probs_str = " ".join([f"{p:.2f}" for p in y_probs_list[j]])
        axes[1, j].set_title(f"Pred (argmax={chosen_ids[j]})\ny=[{probs_str}]")

    plt.suptitle(f"Combined reconstruction | Sample {sample_idx} | Epoch {epoch}")
    plt.tight_layout()
    wandb.log({f"Train combined sequence GT_vs_Pred (E{epoch})": wandb.Image(fig)})
    plt.close(fig)

def log_t_series_panel(plot_targets, plot_preds, y_all, epoch, iteration, max_samples=6, every_k_epochs=5):
    if epoch % every_k_epochs != 0:
        return
    if len(plot_targets) == 0:
        return

    T = len(plot_targets)
    print(f"Logging t-series panel for epoch {epoch}, iteration {iteration}, T={T}")
    Kshow = min(max_samples, plot_targets[0].size(0))
    for sample_idx in range(Kshow):
        fig, axes = plt.subplots(2, T, figsize=(3*T, 6))
        for t in range(T):
            target = plot_targets[t][sample_idx].permute(1, 2, 0).numpy()
            pred   = plot_preds[t][sample_idx].permute(1, 2, 0).numpy()
            axes[0, t].imshow(np.clip(target, 0, 1)); axes[0, t].axis("off")
            axes[1, t].imshow(np.clip(pred,   0, 1)); axes[1, t].axis("off")

            if t == 0 and t < len(y_all):
                y_vec = y_all[t][sample_idx] 
                chosen = torch.argmax(y_vec).item()
                y_soft = " ".join([f"{val:.2f}" for val in y_vec.tolist()])
                axes[0, t].set_title("GT")
                axes[1, t].set_title(f"Pred (argmax={chosen})\ny=[{y_soft}]")

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
    percep_loss_fn = VGGPerceptualLoss().to(device) 

    for i, (data, index) in enumerate(data_loader):
        iteration += 1
        global_step_tracker[0] = iteration

        optimizer = optimizer_spike if iteration < stage_transition_iter else optimizer_full
        optimizer.zero_grad()

        (seq, kp_seq, kp_mask, context_chunk) = data
        seq = seq.to(device)
        context_chunk = context_chunk.to(device)
        kp_seq = kp_seq.to(device)

        # encode first frame
        x0 = seq[:, 0]
        kp0 = kp_seq[:, 0]
        recon_x0, mu0, logvar0, z = generator(x0)
        first_frame_rec = vgg_vae_loss_fn(recon_x0, x0)
        wandb.log({"VAE Loss (first frame)": first_frame_rec.item(),
                   "Iteration": iteration, "Epoch": epoch})

        loss = torch.zeros((), device=seq.device)
        loss += first_frame_rec

        if i == 0:
            log_latent_traversals(
                generator=generator,
                reconstructor=reconstructor,
                latent_vector=latent_vector,
                seq=seq,
                kp_seq=kp_seq,
                epoch=epoch,
                num_vector=num_vector,
                num_steps=num_steps,
                sample_idx=0,
                every_k_epochs=5,
                traversal_alphas=(0,1,2,3,4,5)
            )

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

            with torch.no_grad():
                _, mu_t,  _, _ = generator(x_t)
                _, mu_t1, _, _ = generator(x_t1)

            deltas_bkd = torch.stack(
                [latent_vector[k](mu_t, kp_t1) for k in range(num_vector)],
                dim=1
            )

            selected_delta = (y.unsqueeze(-1) * deltas_bkd).sum(dim=1)   # [B, D]
            z_next_pred    = mu_t + selected_delta
            pred_t1        = generator.inference(z_next_pred)

            L_rec = vgg_vae_loss_fn(pred_t1, x_t1)
            L_lat = F.mse_loss(z_next_pred, mu_t1)
            loss  = loss + L_rec + L_lat

            #perceptual diversity between two random single-flow recons
            if num_vector > 1:
                i_rand = torch.randint(0, num_vector, (1,), device=mu_t.device).item()
                j_rand = (i_rand + torch.randint(1, num_vector, (1,), device=mu_t.device).item()) % num_vector
                recon_i = generator.inference(mu_t + deltas_bkd[:, i_rand, :])
                recon_j = generator.inference(mu_t + deltas_bkd[:, j_rand, :])
                fi = F.normalize(percep_loss_fn.extract_features(recon_i).flatten(1), dim=1, eps=1e-6)
                fj = F.normalize(percep_loss_fn.extract_features(recon_j).flatten(1), dim=1, eps=1e-6)
                dist = 1.0 - (fi * fj).sum(1).clamp(-1+1e-6, 1-1e-6)
                pair_div = F.relu(0.30 - dist).mean()
                loss = loss + pair_div

            if i == 0:
                plot_targets.append(x_t1.detach().cpu())
                plot_preds.append(pred_t1.detach().cpu())

            # target_prob = target_prob + intial_prob
            # intial_prob = (intial_prob * (1-init_switch_prob) + (1 - intial_prob) * init_switch_prob) + torch_binom(
            #     torch.FloatTensor([num_vector]).to(z),
            #     torch.FloatTensor([intial_prob * num_vector]).to(z)
            #     ) * (init_switch_prob ** (intial_prob * num_vector)) * ((1-init_switch_prob) ** (num_vector - intial_prob * num_vector)) * (
            #      1. / num_vector + 2 * init_switch_prob * (1 - init_switch_prob) *
            #      1. / num_vector + (init_switch_prob**2) * 1. / num_vector)

        y_set = torch.cat(y_all, dim=0)   # [T*B, K]
        #bernoulli KL to fixed prior sparsity term
        #encourages ~m active flows per step on average
        eps = 1e-8
        K = y_set.size(1)
        m_target = 1.5              #1-2 flows active
        p = torch.full_like(y_set, m_target / K)  #prior on-prob per flow
        y_clip = y_set.clamp(eps, 1 - eps)
        KL_sparse = (
            y_clip * (y_clip / p.clamp(eps, 1 - eps)).log() +
            (1 - y_clip) * (((1 - y_clip) / (1 - p).clamp(eps, 1 - eps)).log())
        ).mean()
        loss = loss + KL_sparse


        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            gen_enc + gen_dec + gate + list(latent_vector.parameters()), max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

        if i == 0:
            log_t_series_panel(plot_targets, plot_preds, y_all, epoch, iteration, max_samples=6, every_k_epochs=5)
        if iteration % print_freq == 0:
            wandb.log({"Train Loss": loss.item(), "Iteration": iteration, "Epoch": epoch})
            print(f"Epoch [{epoch}/{total_epochs}], Iteration {iteration}: Loss = {loss.item():.4f}")
            wandb.log({"spike_mean": y.mean().item()})
        if iteration % 10 == 0:
            for j in range(num_vector):
                wandb.log({f"spike_usage/flow_{j}": y[:, j].float().mean().item()})

    avg_loss = total_loss / len(data_loader)
    wandb.log({"Epoch Average Loss": avg_loss, "Epoch": epoch})
    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss

@torch.no_grad()
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

    alphas = torch.linspace(0, 6, steps=7, device=device)
    num_samples_to_log = 6

    for i, (data, index) in enumerate(data_loader):
        (seq, kp_seq, kp_mask, context_chunk) = data
        seq = seq.to(device)
        context_chunk = context_chunk.to(device)
        kp_seq = kp_seq.to(device)

        x0 = seq[:, 0]
        kp0 = kp_seq[:, 0]

        recon_x0, mu0, logvar0, z0 = generator(x0)
        first_rec = loss_fn(recon_x0, x0).item()
        total_loss += first_rec

        x_t1 = x0
        kp_t1 = kp0

        for t in range(1, num_steps + 1):
            x_t = x_t1
            x_t1 = seq[:, t]
            kp_t = kp_t1
            kp_t1 = kp_seq[:, t]

            x_pair = torch.cat([x_t, x_t1, x_t1-x_t], dim=1)

            y, g = reconstructor(x_pair, iter=None)
            _, mu_t,  _, _ = generator(x_t)
            _, mu_t1, _, _ = generator(x_t1)

            deltas_bkd = torch.stack(
                [latent_vector[k](mu_t, kp_t1) for k in range(num_vector)],
                dim=1
            ) 

            selected_delta = (y.unsqueeze(-1) * deltas_bkd).sum(dim=1)  # [B,D]
            z_next_pred    = mu_t + selected_delta

            pred_t1    = generator.inference(z_next_pred)
            L_rec_step = loss_fn(pred_t1, x_t1).item()
            L_lat_step = F.mse_loss(z_next_pred, mu_t1).item()
            total_loss += (L_rec_step + L_lat_step)

        num_batches += 1
        break

    avg_loss = total_loss / max(num_batches, 1)
    wandb.log({
        "Validation Loss": avg_loss,
        "Epoch": epoch,
        "Val Stage": "slab" if not spike_only else "spike",
        "Val FirstFrame Recon": first_rec,
    })
    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss


# --- Training setup ---
total_epochs = 300
batch_size = 8
wandb.init(project="latent_flow", config={"epochs": total_epochs, "batch_size": batch_size})

# Setup VAE + reconstructor
num_vector = 4
latent_dim = 256
kp_dim = 2*7
img_size = 128
kp_window = 5
cond_dim = (2*kp_window + 1) * 14
flow_dim = 96
flow_hidden_dim = 256

generator = DinoUNetVAE(embed_dim=latent_dim, img_size=img_size)
reconstructor = CNNTransformerReconstructor(img_size=img_size, in_channels=3*3, latent_size=num_vector) #uses gumbel sigmoid 
latent_vector = torch.nn.ModuleList([LatentFieldWithKeypoints(latent_dim=latent_dim, kp_dim=cond_dim, hidden_dim=flow_hidden_dim, flow_dim=flow_dim) for _ in range(num_vector)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("ae_pretrained_128.pt", map_location=device)
generator.load_state_dict(checkpoint)
generator.to(device); reconstructor.to(device); latent_vector.to(device)
percep_loss_fn = VGGPerceptualLoss().to(device)

#optimizers 
gen_enc = list(generator.encoder.parameters())
gen_dec = list(generator.decoder.parameters())
gate    = list(reconstructor.parameters())
flows   = list(latent_vector.parameters())
optimizer_spike = optim.AdamW([
    {"params": gen_enc, "lr": 1e-5},
    {"params": gen_dec, "lr": 5e-5},
    {"params": gate,    "lr": 2e-5},
    {"params": flows,   "lr": 5e-5},
], betas=(0.9, 0.999), weight_decay=1e-4)

optimizer_full = optim.AdamW([
    {"params": gen_enc, "lr": 5e-6},
    {"params": gen_dec, "lr": 5e-5},
    {"params": gate,    "lr": 1e-5},
    {"params": flows,   "lr": 8e-5},
], betas=(0.9, 0.999), weight_decay=1e-4)

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
train_size = int(0.8 * len(video_dataset))
val_size = len(video_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(video_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

stage_transition_iter = 20000

# Training loop
for epoch in range(1, total_epochs + 1):
    print(f"--- Epoch {epoch}/{total_epochs} ---")
    training_loss = training_function(train_loader, optimizer_spike, optimizer_full, generator, reconstructor, latent_vector,
                                      num_steps=6, num_vector=num_vector, epoch=epoch, total_epochs=total_epochs,
                                      stage_transition_iter=stage_transition_iter, global_step_tracker=global_step_tracker)
    validation_loss = validation_function(val_loader, generator, reconstructor, latent_vector, num_steps=6,
                                      num_vector=num_vector, epoch=epoch, loss_fn=vgg_vae_loss_fn,
                                      stage_transition_iter=stage_transition_iter, global_step_tracker=global_step_tracker)
