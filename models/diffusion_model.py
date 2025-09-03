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

# ================= 自适应层归一化 (AdaLN) 模块 =================
class AdaLN(nn.Module):
    def __init__(self, num_features, cond_feat_dim):
        super().__init__()
        # 使用一个线性层从条件特征中预测出缩放(scale)和偏移(bias)参数
        self.projection = nn.Linear(cond_feat_dim, num_features * 2)
        self.norm = nn.InstanceNorm1d(num_features) # 使用InstanceNorm更适合风格迁移

    def forward(self, x, cond_feat):
        # x: [B, C, N], cond_feat: [B, D]
        params = self.projection(cond_feat).unsqueeze(-1) # -> [B, C*2, 1]
        scale, bias = torch.chunk(params, 2, dim=1) # -> [B, C, 1], [B, C, 1]
        
        # 应用归一化和仿射变换
        return self.norm(x) * (scale + 1) + bias
# ===================================================================

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

class GlobalContextExtractor(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.encoder = PointNet2Encoder(input_channels=3, feature_dim=feature_dim)
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.encoder(points)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 time_emb_dim: int, cond_feat_dim: int):
        super().__init__()
        self.mlp_time = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        
        # ================= 使用AdaLN替换旧的条件注入方式 =================
        self.norm1 = AdaLN(out_channels, cond_feat_dim)
        self.norm2 = nn.BatchNorm1d(out_channels) # 第二个norm保持不变
        # =====================================================================
        
        self.silu = nn.SiLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        
        # ================= 应用AdaLN和时间嵌入 =================
        h = self.norm1(h, cond_feat)
        h = h + self.mlp_time(time_emb).unsqueeze(-1)
        h = self.silu(h)
        # ==========================================================
        
        h = self.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)

class UNetBackbone(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.init_conv = nn.Conv1d(3, 128, 1)
        
        self.down1 = ResidualBlock(128, 256, config.time_embed_dim, config.feature_dim)
        self.pool1 = nn.MaxPool1d(2)
        self.down2 = ResidualBlock(256, 512, config.time_embed_dim, config.feature_dim)
        self.pool2 = nn.MaxPool1d(2)
        
        self.mid = ResidualBlock(512, 512, config.time_embed_dim, config.feature_dim)
        
        self.up1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1_resblock = ResidualBlock(512 + 512, 256, config.time_embed_dim, config.feature_dim)
        
        self.up2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_resblock = ResidualBlock(256 + 256, 128, config.time_embed_dim, config.feature_dim)
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.SiLU(),
            nn.Conv1d(128, 3, 1)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        
        h1 = self.init_conv(x)
        h2 = self.down1(h1, time_emb, cond_feat)
        p1 = self.pool1(h2)
        
        h3 = self.down2(p1, time_emb, cond_feat)
        p2 = self.pool2(h3)
        
        h_mid = self.mid(p2, time_emb, cond_feat)
        
        h_up1 = self.up1_upsample(h_mid)
        h_up1_in = torch.cat([h_up1, h3], dim=1)
        h_up1_out = self.up1_resblock(h_up1_in, time_emb, cond_feat)
        
        h_up2 = self.up2_upsample(h_up1_out)
        h_up2_in = torch.cat([h_up2, h2], dim=1)
        h_up2_out = self.up2_resblock(h_up2_in, time_emb, cond_feat)
        
        output = self.final_conv(h_up2_out)
        return output.permute(0, 2, 1)

class LocalRefinementNetwork(nn.Module):
    """局部细节恢复网络"""
    def __init__(self, config: Config):
        super().__init__()
        self.chunk_size = config.chunk_size
        
        # 局部特征提取
        self.local_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        
        # 细节生成
        self.detail_generator = nn.Sequential(
            nn.Conv1d(256 + 3, 128, 1),  # 特征 + 粗糙点
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)  # 输出残差
        )
        
    def forward(self, coarse_points: torch.Tensor, 
                original_structure: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_points: [B, N, 3] - 转换后的粗糙点云
            original_structure: [B, N, 3] - 原始结构信息
        Returns:
            refined_points: [B, N, 3] - 细化后的点云
        """
        B, N, _ = coarse_points.shape
        
        # 提取局部特征
        local_feat = self.local_encoder(original_structure.permute(0, 2, 1))  # [B, 256, N]
        
        # 拼接特征和粗糙点
        combined = torch.cat([local_feat, coarse_points.permute(0, 2, 1)], dim=1)  # [B, 259, N]
        
        # 生成细节残差
        residual = self.detail_generator(combined).permute(0, 2, 1)  # [B, N, 3]
        
        # 添加残差
        refined_points = coarse_points + residual * 0.1  # 缩放残差避免过大变化
        
        return refined_points

class PointCloudDiffusionModel(nn.Module):
    """分层Diffusion模型主体"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(config.time_embed_dim)
        
        # 全局处理分支（处理下采样的点云）
        self.global_encoder = GlobalContextExtractor(feature_dim=config.feature_dim)
        self.global_backbone = UNetBackbone(config)
        
        # 局部细化分支
        self.local_refinement = LocalRefinementNetwork(config)
        
        # 下采样目标大小
        self.downsample_size = 30000
        
    def voxel_downsample(self, points: torch.Tensor, target_size: int = 30000) -> Tuple[torch.Tensor, Dict]:
        """体素下采样
        Args:
            points: [B, N, 3] - 原始点云
            target_size: 目标点数
        Returns:
            downsampled: [B, target_size, 3] - 下采样点云
            info: 采样信息字典
        """
        
        with torch.no_grad():
            B, N, _ = points.shape
            device = points.device
            
            downsampled_list = []
            indices_list = []
            
            for b in range(B):
                pts = points[b].cpu().numpy()
                
                # 计算体素大小
                pts_min = pts.min(axis=0)
                pts_max = pts.max(axis=0)
                pts_range = pts_max - pts_min
                voxel_size = (pts_range.prod() / target_size) ** (1/3) * 1.5
                
                # 体素化
                voxel_dict = {}
                for i, pt in enumerate(pts):
                    voxel_key = tuple((pt / voxel_size).astype(int))
                    if voxel_key not in voxel_dict:
                        voxel_dict[voxel_key] = []
                    voxel_dict[voxel_key].append(i)
                
                # 从每个体素选择代表点
                selected_indices = []
                for indices in voxel_dict.values():
                    voxel_points = pts[indices]
                    center = voxel_points.mean(axis=0)
                    distances = np.linalg.norm(voxel_points - center, axis=1)
                    selected_idx = indices[np.argmin(distances)]
                    selected_indices.append(selected_idx)
                
                if len(selected_indices) < target_size:
                    remaining = target_size - len(selected_indices)
                    all_indices = set(range(N))
                    available = list(all_indices - set(selected_indices))
                    if available:
                        extra = np.random.choice(available, min(remaining, len(available)), replace=False)
                        selected_indices.extend(extra)
                
                if len(selected_indices) > target_size:
                    selected_indices = np.random.choice(selected_indices, target_size, replace=False)
                
                selected_indices = np.array(selected_indices)
                downsampled_pts = pts[selected_indices]
                
                downsampled_list.append(torch.from_numpy(downsampled_pts).float())
                indices_list.append(selected_indices)
            
            downsampled = torch.stack(downsampled_list).to(device)
            
            return downsampled, {'indices': indices_list}
    
    def forward(self, noisy_points: torch.Tensor, time: torch.Tensor, 
                condition_points: torch.Tensor, use_hierarchical: bool = True) -> torch.Tensor:
        """
        前向传播
        """
        if not use_hierarchical or noisy_points.shape[1] <= self.downsample_size:
            time_emb = self.time_embedding(time)
            cond_feat = self.global_encoder(condition_points)
            return self.global_backbone(noisy_points, time_emb, cond_feat)
        
        # 分层处理
        # 1. 下采样
        noisy_down, down_info = self.voxel_downsample(noisy_points, self.downsample_size)
        cond_down, _ = self.voxel_downsample(condition_points, self.downsample_size)
        
        # 2. 全局处理
        time_emb = self.time_embedding(time)
        cond_feat = self.global_encoder(cond_down)
        denoised_coarse = self.global_backbone(noisy_down, time_emb, cond_feat)
        
        # 3. 上采样到原始大小
        denoised_full = self.upsample_points(denoised_coarse, noisy_points, down_info)
        
        # 4. 局部细化
        refined = self.local_refinement(denoised_full, noisy_points)
        
        return refined
    
    def upsample_points(self, coarse_points: torch.Tensor, 
                        original_points: torch.Tensor, 
                        down_info: Dict) -> torch.Tensor:
        
        """将粗糙点云上采样到原始大小"""
        B, N, _ = original_points.shape
        device = coarse_points.device
        upsampled = torch.zeros_like(original_points)
        
        for b in range(B):
            indices = down_info['indices'][b]
            
            # 使用 .detach() 来断开梯度连接
            coarse_b = coarse_points[b].cpu().detach().numpy()
            original_b = original_points[b].cpu().detach().numpy()
            
            result = np.zeros_like(original_b)
            result[indices] = coarse_b
            
            all_indices = set(range(N))
            unsampled = list(all_indices - set(indices))
            
            if unsampled:
                nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(original_b[indices])
                distances, neighbors = nbrs.kneighbors(original_b[unsampled])
                
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                
                # 获取 coarse_b 中对应的邻居点
                interpolated_points = np.sum(coarse_b[neighbors] * weights[:, :, np.newaxis], axis=1)
                result[unsampled] = interpolated_points

            upsampled[b] = torch.from_numpy(result).float().to(device)
        
        return upsampled

class DiffusionProcess:
    """Diffusion过程管理器"""
    def __init__(self, config: Config, device: str = 'cuda'):
        self.num_timesteps = config.num_timesteps
        self.device = device
        self.betas = self._get_beta_schedule(config.beta_schedule, config.noise_schedule_offset).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

    def _get_beta_schedule(self, schedule_name: str, offset: float = 0.0) -> torch.Tensor:
        if schedule_name == "linear":
            betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        elif schedule_name == "cosine":
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008 + offset) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
        return betas

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        noisy_points = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_points, noise

    @torch.no_grad()
    def ddim_sample_loop(self, model: PointCloudDiffusionModel, shape: Tuple, 
                         condition_points: torch.Tensor, num_inference_steps: int = 50):
        device = torch.device(self.device)
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        use_hierarchical = shape[1] > 30000
        
        for i in tqdm(range(len(timesteps)), desc="DDIM Sampling"):
            t = timesteps[i]
            prev_t = timesteps[i+1] if i < len(timesteps) - 1 else torch.tensor(-1, device=device)
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x, batch_t, condition_points, use_hierarchical=use_hierarchical)
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
            
            pred_x0 = (x - torch.sqrt(1. - alpha_t) * predicted_noise) / (torch.sqrt(alpha_t) + 1e-8)
            
            norm = torch.linalg.norm(pred_x0, dim=-1, keepdim=True)
            pred_x0 = pred_x0 / torch.maximum(norm, torch.tensor(1.0, device=pred_x0.device))
            
            dir_xt = torch.sqrt(1. - alpha_t_prev) * predicted_noise
            x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
            x = x_prev
        return x

    @torch.no_grad()
    def guided_sample_loop(self, model: PointCloudDiffusionModel, 
                          source_points: torch.Tensor, condition_points: torch.Tensor, 
                          num_inference_steps: int = 50, guidance_strength: float = 0.5):
        device = torch.device(self.device)
        shape = source_points.shape
        start_timestep = int((1.0 - guidance_strength) * (self.num_timesteps - 1))
        start_timestep = max(0, min(start_timestep, self.num_timesteps - 1))
        t_start = torch.tensor([start_timestep], device=device).long()
        noise = torch.randn_like(source_points)
        x, _ = self.q_sample(source_points, t_start, noise)
        timesteps = torch.linspace(start_timestep, 0, num_inference_steps).long().to(device)
        
        use_hierarchical = shape[1] > 30000
        
        for i in tqdm(range(len(timesteps)), desc="Guided Sampling"):
            t = timesteps[i]
            prev_t = timesteps[i+1] if i < len(timesteps) - 1 else torch.tensor(-1, device=device)
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x, batch_t, condition_points, use_hierarchical=use_hierarchical)
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
            
            pred_x0 = (x - torch.sqrt(1. - alpha_t) * predicted_noise) / (torch.sqrt(alpha_t) + 1e-8)
            
            norm = torch.linalg.norm(pred_x0, dim=-1, keepdim=True)
            pred_x0 = pred_x0 / torch.maximum(norm, torch.tensor(1.0, device=pred_x0.device))
            
            dir_xt = torch.sqrt(1. - alpha_t_prev) * predicted_noise
            x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
            x = x_prev
        return x