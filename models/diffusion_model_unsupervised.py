import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import math
from tqdm import tqdm


class TimeEmbedding(nn.Module):
    """时间嵌入模块"""
    
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


class ResidualBlock(nn.Module):
    """残差块with时间条件"""
    
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, out_channels),
            nn.SiLU()
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.shortcut(x)


class StyleEncoder(nn.Module):
    """风格编码器 - 提取域特征"""
    
    def __init__(self, input_dim: int = 3, hidden_dims: List[int] = [64, 128, 256, 512], 
                 style_dim: int = 256):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_dim, h_dim, 1),
                nn.GroupNorm(8, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.encoder(x)
        avg_pool = features.mean(dim=2)
        max_pool = features.max(dim=2)[0]
        global_features = (avg_pool + max_pool) / 2
        style = self.style_head(global_features)
        return style


class ContentEncoder(nn.Module):
    """内容编码器 - 保留几何结构"""
    
    def __init__(self, input_dim: int = 3, hidden_dims: List[int] = [64, 128, 256]):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dims[0], 1)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], 1)
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(hidden_dims[1], hidden_dims[2], 1)
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.relu3 = nn.ReLU()
        self.coord_conv = nn.Conv1d(input_dim, hidden_dims[2], 1)
        self.fusion = nn.Conv1d(hidden_dims[2] * 2, hidden_dims[2], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.relu1(self.bn1(self.conv1(x)))
        h = self.relu2(self.bn2(self.conv2(h)))
        h = self.relu3(self.bn3(self.conv3(h)))
        coord_features = self.coord_conv(x)
        combined = torch.cat([h, coord_features], dim=1)
        content = self.fusion(combined)
        return content


class SpatialAwareStyleModulation(nn.Module):
    """空间感知的风格调制"""
    
    def __init__(self, feature_channels: int, style_dim: int):
        super().__init__()
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, feature_channels * 2),
            nn.ReLU(),
            nn.Linear(feature_channels * 2, feature_channels * 2)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(feature_channels, feature_channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(feature_channels // 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B, C, N = features.shape
        style_params = self.style_mlp(style)
        scale, shift = style_params.chunk(2, dim=1)
        spatial_weight = self.spatial_attention(features)
        mean = features.mean(dim=2, keepdim=True)
        std = features.std(dim=2, keepdim=True) + 1e-5
        normalized = (features - mean) / std
        scale = scale[:, :, None]
        shift = shift[:, :, None]
        modulated = normalized * (1 + scale * spatial_weight * 0.1) + shift * spatial_weight * 0.1
        alpha = 0.1
        output = features * (1 - alpha) + modulated * alpha
        return output


class UnsupervisedPointCloudDiffusionModel(nn.Module):
    """无监督点云Diffusion模型"""
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: List[int] = [128, 256, 512, 1024],
                 time_dim: int = 256,
                 style_dim: int = 256,
                 content_dims: List[int] = [64, 128, 256]):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.style_encoder = StyleEncoder(input_dim, style_dim=style_dim)
        self.content_encoder = ContentEncoder(input_dim, hidden_dims=content_dims)
        content_dim = content_dims[-1]
        self.input_proj = nn.Conv1d(input_dim + content_dim, hidden_dims[0], 1)
        self.style_modulation = nn.ModuleList([
            SpatialAwareStyleModulation(h_dim, style_dim) for h_dim in hidden_dims
        ])
        self.middle_style_modulation = SpatialAwareStyleModulation(hidden_dims[-1], style_dim)
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[i], hidden_dims[i+1], time_dim)
            for i in range(len(hidden_dims) - 1)
        ])
        self.middle_block = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_dim)
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-(i+1)] * 2, hidden_dims[-(i+2)], time_dim)
            for i in range(len(hidden_dims) - 1)
        ])
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dims[0], hidden_dims[0] // 2, 1), nn.SiLU(),
            nn.Conv1d(hidden_dims[0] // 2, input_dim, 1)
        )
        self.input_skip = nn.Conv1d(input_dim, input_dim, 1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
            style_condition: Optional[torch.Tensor] = None,
            content_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        time_emb = self.time_embed(t)
        if style_condition is None: style_condition = self.style_encoder(x)
        if content_condition is None: content_condition = self.content_encoder(x)
        x_input = x.transpose(1, 2)
        x = torch.cat([x_input, content_condition], dim=1)
        h = self.input_proj(x)
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, time_emb)
            if style_condition is not None:
                h = self.style_modulation[i+1](h, style_condition)
            skip_connections.append(h)
        h = self.middle_block(h, time_emb)
        if style_condition is not None: h = self.middle_style_modulation(h, style_condition)
        for i, block in enumerate(self.decoder_blocks):
            h = torch.cat([h, skip_connections[-(i+1)]], dim=1)
            h = block(h, time_emb)
        out = self.output_proj(h)
        out = out + self.input_skip(x_input) * 0.5
        return out.transpose(1, 2)


class UnsupervisedDiffusionProcess:
    """无监督Diffusion过程"""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_schedule: str = "cosine",
                 device: str = "cuda"):
        self.num_timesteps = num_timesteps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps).to(self.device)
        else:
            self.betas = self._linear_beta_schedule(num_timesteps).to(self.device)
        
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _linear_beta_schedule(self, timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32)**2
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(x_start.shape[0], 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(x_start.shape[0], 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...], 
            style_condition: Optional[torch.Tensor] = None,
            content_condition: Optional[torch.Tensor] = None,
            num_inference_steps: Optional[int] = 50) -> torch.Tensor:
        """
        生成采样 (DDIM)，并加入关键的clamp操作来保证数值稳定。
        """
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for i in tqdm(range(len(timesteps)), desc="DDIM Sampling", leave=False):
            t = timesteps[i]
            prev_t = timesteps[i+1] if i < len(timesteps) - 1 else torch.tensor(-1, device=device)
            
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x, batch_t, style_condition, content_condition)
            
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
            
            # 预测x0 (去噪后的原始点云)
            pred_x0 = (x - torch.sqrt(1. - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # ======================================================================
            #               !!! 核心修复：添加“护栏” !!!
            # 这是解决数值爆炸和巨大val_loss的“一针见血”的办法。
            # 它强制模型在每一步的预测都必须在合理的坐标范围内，
            # 从而阻止了误差的雪崩式累积。
            # ======================================================================
            pred_x0 = torch.clamp(pred_x0, -1.5, 1.5)
            
            # 计算方向导数
            dir_xt = torch.sqrt(1. - alpha_t_prev) * predicted_noise
            
            # 更新x_t -> x_{t-1}
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return x
