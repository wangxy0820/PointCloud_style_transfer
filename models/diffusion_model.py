# models/diffusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .pointnet2_encoder import PointNet2Encoder
from config.config import Config

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class StyleEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.encoder = PointNet2Encoder(input_channels=3, feature_dim=feature_dim)
        self.style_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, feature_dim), nn.ReLU())
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.style_mlp(self.encoder(points))

class NoisePredictor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, config.feature_dim))
        self.time_embedding = TimeEmbedding(config.time_embed_dim)
        self.time_proj = nn.Linear(config.time_embed_dim, config.feature_dim)
        self.style_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(config.feature_dim, config.feature_dim * 2), nn.ReLU(),
                          nn.Linear(config.feature_dim * 2, config.feature_dim), nn.Dropout(0.1))
            for _ in range(6)])
        self.output_mlp = nn.Sequential(
            nn.Linear(config.feature_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, noisy_points: torch.Tensor, timestep: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        point_feat = self.point_encoder(noisy_points)
        time_feat = self.time_proj(self.time_embedding(timestep)).unsqueeze(1)
        style_feat_proj = self.style_proj(style_feat).unsqueeze(1)
        x = point_feat + time_feat + style_feat_proj
        for layer in self.layers:
            x = layer(x) + x
        return self.output_mlp(x)


class HierarchicalProcessor:
    def __init__(self, total_points: int = 120000, global_points: int = 30000):
        self.total_points = total_points
        self.global_points = global_points

    def _voxel_grid_downsample_torch(self, points: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if points.shape[1] <= target_size:
            indices = torch.arange(points.shape[1], device=points.device)
            return points, indices.expand(points.shape[0], -1)

        B, N, _ = points.shape
        downsampled_list = []
        indices_list = []

        for b in range(B):
            pts = points[b]
            xyz_min = pts.min(axis=0)[0]
            xyz_max = pts.max(axis=0)[0]
            xyz_range = xyz_max - xyz_min
            xyz_range[xyz_range < 1e-6] = 1.0

            voxel_size = (xyz_range.prod() / target_size)**(1/3) * 1.2
            if voxel_size < 1e-6:
                voxel_size = 1e-3

            voxel_indices = torch.floor((pts - xyz_min) / voxel_size).int()
            voxel_hash = (voxel_indices[:, 0] * 73856093) ^ (voxel_indices[:, 1] * 19349663) ^ (voxel_indices[:, 2] * 83492791)
            
            unique_voxels, inverse_indices = torch.unique(voxel_hash, return_inverse=True, return_counts=False)
            
            perm = torch.arange(unique_voxels.size(0), dtype=torch.long, device=pts.device)
            representative_indices = torch.zeros(unique_voxels.size(0), dtype=torch.long, device=pts.device)
            representative_indices.scatter_add_(0, inverse_indices, torch.arange(N, dtype=torch.long, device=pts.device))
            representative_indices = (representative_indices / torch.bincount(inverse_indices)).long()

            current_size = len(representative_indices)
            if current_size > target_size:
                rand_indices = torch.randperm(current_size, device=pts.device)[:target_size]
                final_indices = representative_indices[rand_indices]
            elif current_size < target_size:
                remaining_needed = target_size - current_size
                all_indices = torch.arange(N, device=pts.device)
                mask = torch.ones(N, dtype=torch.bool, device=pts.device)
                mask[representative_indices] = False
                pool = all_indices[mask]
                
                if len(pool) > 0:
                    num_to_sample = min(remaining_needed, len(pool))
                    additional_indices = pool[torch.randperm(len(pool), device=pts.device)[:num_to_sample]]
                    final_indices = torch.cat([representative_indices, additional_indices])
                else:
                    final_indices = representative_indices
            else:
                final_indices = representative_indices

            downsampled_list.append(pts[final_indices])
            indices_list.append(final_indices)

        return torch.stack(downsampled_list), torch.stack(indices_list)

    def downsample(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._voxel_grid_downsample_torch(points, self.global_points)

    def upsample_knn(self, coarse_points: torch.Tensor, original_points: torch.Tensor, 
                     coarse_indices: torch.Tensor) -> torch.Tensor:
        B, N_orig, _ = original_points.shape
        device = coarse_points.device
        upsampled_list = []
        for b in range(B):
            coarse_b_np = coarse_points[b].cpu().detach().numpy()
            original_b_np = original_points[b].cpu().detach().numpy()
            indices_b_np = coarse_indices[b].cpu().detach().numpy()
            result = np.zeros_like(original_b_np)
            valid_indices = indices_b_np[indices_b_np < N_orig]
            valid_coarse_points = coarse_b_np[:len(valid_indices)]
            result[valid_indices] = valid_coarse_points
            unknown_mask = np.ones(N_orig, dtype=bool)
            unknown_mask[valid_indices] = False
            unknown_indices = np.where(unknown_mask)[0]
            if len(unknown_indices) > 0 and len(valid_indices) > 0:
                k = min(3, len(valid_indices))
                fit_points = original_b_np[valid_indices]
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(fit_points)
                distances, neighbors = nbrs.kneighbors(original_b_np[unknown_indices])
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                interpolated = np.sum(valid_coarse_points[neighbors] * weights[..., np.newaxis], axis=1)
                result[unknown_indices] = interpolated
            upsampled_list.append(torch.from_numpy(result).float().to(device))
        return torch.stack(upsampled_list)


class PointCloudDiffusionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.style_encoder = StyleEncoder(feature_dim=config.feature_dim)
        self.noise_predictor = NoisePredictor(config)
        self.hierarchical_processor = HierarchicalProcessor(
            total_points=config.total_points, global_points=config.global_points)
        
    def forward(self, noisy_points: torch.Tensor, timestep: torch.Tensor, 
                condition_points: torch.Tensor, cond_drop_prob: float = 0.0,
                use_hierarchical: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # Removed `with torch.no_grad():` to allow style_encoder to train
        if use_hierarchical and condition_points.shape[1] > self.config.global_points:
            cond_downsampled, _ = self.hierarchical_processor.downsample(condition_points)
            style_feat = self.style_encoder(cond_downsampled)
        else:
            style_feat = self.style_encoder(condition_points)
        
        if cond_drop_prob > 0:
            mask = torch.rand(style_feat.shape[0], 1, device=style_feat.device) > cond_drop_prob
            style_feat = style_feat * mask
        
        if use_hierarchical and noisy_points.shape[1] > self.config.global_points:
            # Hierarchical path
            noisy_downsampled, noise_indices = self.hierarchical_processor.downsample(noisy_points)
            predicted_noise_coarse = self.noise_predictor(noisy_downsampled, timestep, style_feat)
            # Return the coarse prediction and indices for loss calculation
            return predicted_noise_coarse, noise_indices
        else:
            # Direct path
            predicted_noise = self.noise_predictor(noisy_points, timestep, style_feat)
            # Return the full prediction and None for indices
            return predicted_noise, None


class DiffusionProcess:
    def __init__(self, config: Config, device: str = 'cuda'):
        self.num_timesteps = config.num_timesteps
        self.device = device
        self.betas = self._get_beta_schedule(config.beta_schedule, config.noise_schedule_offset).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def _get_beta_schedule(self, schedule_name: str, offset: float = 0.0) -> torch.Tensor:
        if schedule_name == "cosine":
            steps = self.num_timesteps + 1; x = torch.linspace(0, self.num_timesteps, steps, device=self.device)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008 + offset) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]); return torch.clip(betas, 0.0001, 0.9999)
        elif schedule_name == "linear":
            return torch.linspace(0.0001, 0.02, self.num_timesteps, device=self.device)
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None: noise = torch.randn_like(x_start)
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def _apply_geometric_constraints(self, points: torch.Tensor, target_range: float = 1.8) -> torch.Tensor:
        return torch.tanh(points / target_range) * target_range

    @torch.no_grad()
    def guided_sample_loop(self, model: PointCloudDiffusionModel, 
                          source_points: torch.Tensor, condition_points: torch.Tensor,
                          num_inference_steps: int = 50, guidance_scale: float = 7.5) -> torch.Tensor:
        device = source_points.device; shape = source_points.shape; B = shape[0]
        
        with torch.no_grad():
            style_feat = model.style_encoder(model.hierarchical_processor.downsample(condition_points)[0])
        uncond_style_feat = torch.zeros_like(style_feat)
        
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps).long().to(device)
        use_hierarchical = shape[1] > model.config.global_points

        for t in tqdm(timesteps, desc="Guided Style Transfer (CFG)"):
            batch_t = torch.full((B,), t, device=device, dtype=torch.long)
            x_in = torch.cat([x] * 2); t_in = torch.cat([batch_t] * 2)
            style_in = torch.cat([style_feat, uncond_style_feat])
            
            # Since inference requires the upsampled result, we do the full hierarchical process
            x_coarse, x_indices = model.hierarchical_processor.downsample(x_in)
            noise_coarse = model.noise_predictor(x_coarse, t_in, style_in)
            predicted_noise_both = model.hierarchical_processor.upsample_knn(noise_coarse, x_in, x_indices)

            noise_pred_cond, noise_pred_uncond = predicted_noise_both.chunk(2)
            final_predicted_noise = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            alpha_t = self.alphas_cumprod[t]
            t_prev_val = timesteps[timesteps.tolist().index(t) + 1] if t > 0 else -1
            alpha_t_prev = self.alphas_cumprod[t_prev_val] if t_prev_val >= 0 else torch.tensor(1.0, device=device)
            
            pred_x0 = (x - torch.sqrt(1. - alpha_t) * final_predicted_noise) / (torch.sqrt(alpha_t) + 1e-8)
            pred_x0 = pred_x0 + 0.1 * (source_points - pred_x0)
            pred_x0 = self._apply_geometric_constraints(pred_x0)
            
            dir_xt = torch.sqrt(1. - alpha_t_prev) * final_predicted_noise
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        return x

    @torch.no_grad()
    def ddim_sample_loop(self, model: PointCloudDiffusionModel, shape: Tuple, 
                         condition_points: torch.Tensor, num_inference_steps: int = 50) -> torch.Tensor:
        device = condition_points.device
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        use_hierarchical = shape[1] > model.config.global_points
        
        for i in tqdm(range(len(timesteps)), desc="DDIM Sampling"):
            t = timesteps[i]
            prev_t = timesteps[i+1] if i < len(timesteps) - 1 else torch.tensor(-1, device=device)
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # The model's forward pass returns a tuple (noise, indices). We only need the noise for DDIM.
            # In inference, we need the full upsampled noise.
            if use_hierarchical:
                 noise_coarse, indices = model(x, batch_t, condition_points, cond_drop_prob=0, use_hierarchical=True)
                 predicted_noise = model.hierarchical_processor.upsample_knn(noise_coarse, x, indices)
            else:
                 predicted_noise, _ = model(x, batch_t, condition_points, cond_drop_prob=0, use_hierarchical=False)

            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
            
            pred_x0 = (x - torch.sqrt(1. - alpha_t) * predicted_noise) / (torch.sqrt(alpha_t) + 1e-8)
            pred_x0 = self._apply_geometric_constraints(pred_x0)
            
            dir_xt = torch.sqrt(1. - alpha_t_prev) * predicted_noise
            x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
            x = x_prev
        return x