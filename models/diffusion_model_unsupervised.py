# models/unsupervised_diffusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import math


class TimeEmbedding(nn.Module):
    """时间嵌入模块"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        创建正弦时间嵌入
        Args:
            t: 时间步 [B]
        Returns:
            时间嵌入 [B, dim]
        """
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
        """
        Args:
            x: [B, C, N]
            t: [B, time_channels]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # 添加时间信息
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
        
        # 全局池化提取风格
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 3]
        Returns:
            style: [B, style_dim]
        """
        x = x.transpose(1, 2)  # [B, 3, N]
        features = self.encoder(x)  # [B, C, N]
        
        # 全局平均池化和最大池化
        avg_pool = features.mean(dim=2)  # [B, C]
        max_pool = features.max(dim=2)[0]  # [B, C]
        
        # 组合特征
        global_features = (avg_pool + max_pool) / 2
        
        # 提取风格
        style = self.style_head(global_features)
        
        return style


class ContentEncoder(nn.Module):
    """内容编码器 - 保留几何结构 (已修改)"""
    
    def __init__(self, input_dim: int = 3, hidden_dims: List[int] = [64, 128, 256]):
        super().__init__()
        
        # 确保网络产生有意义的特征
        self.conv1 = nn.Conv1d(input_dim, hidden_dims[0], 1)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], 1)
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(hidden_dims[1], hidden_dims[2], 1)
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.relu3 = nn.ReLU()
        
        # 坐标变换分支 - 保持空间信息
        self.coord_conv = nn.Conv1d(input_dim, hidden_dims[2], 1)
        
        # 最终融合
        self.fusion = nn.Conv1d(hidden_dims[2] * 2, hidden_dims[2], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 3]
        Returns:
            content: [B, C, N]
        """
        x = x.transpose(1, 2)  # [B, 3, N]
        
        # 主路径
        h = self.relu1(self.bn1(self.conv1(x)))
        h = self.relu2(self.bn2(self.conv2(h)))
        h = self.relu3(self.bn3(self.conv3(h)))
        
        # 坐标路径 - 直接编码空间信息
        coord_features = self.coord_conv(x)
        
        # 拼接并融合
        combined = torch.cat([h, coord_features], dim=1)
        content = self.fusion(combined)
        
        # REMOVED: 移除了可能导致模型坍塌的全局平均值偏置。
        # 这个操作会污染局部几何特征，使模型倾向于生成平均化的“团状”结构。
        # content = content + 0.1 * x.mean(dim=1, keepdim=True)
        
        return content


class PositionalEncoding(nn.Module):
    """位置编码 - 保持空间信息"""
    
    def __init__(self, d_model: int = 3, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        if d_model >= 2:
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 0:
                pe[:, 1::2] = torch.cos(position * div_term)
            else:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 0] = position.squeeze()
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: int) -> torch.Tensor:
        return self.pe[:x]


class SpatialAwareStyleModulation(nn.Module):
    """空间感知的风格调制 - 不破坏结构 (已修改)"""
    
    def __init__(self, feature_channels: int, style_dim: int):
        super().__init__()
        
        # 生成空间感知的调制参数
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, feature_channels * 2),
            nn.ReLU(),
            nn.Linear(feature_channels * 2, feature_channels * 2)
        )
        
        # 空间注意力 - 决定哪些区域应该被调制
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(feature_channels, feature_channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(feature_channels // 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, N]
            style: [B, style_dim]
        Returns:
            modulated features: [B, C, N]
        """
        B, C, N = features.shape
        
        # 生成调制参数
        style_params = self.style_mlp(style)  # [B, C*2]
        scale, shift = style_params.chunk(2, dim=1)  # 各[B, C]
        
        # 计算空间注意力
        spatial_weight = self.spatial_attention(features)  # [B, 1, N]
        
        # 使用全局统计进行归一化
        mean = features.mean(dim=2, keepdim=True)  # [B, C, 1]
        std = features.std(dim=2, keepdim=True) + 1e-5  # [B, C, 1]
        normalized = (features - mean) / std
        
        # 应用风格调制
        scale = scale[:, :, None]  # [B, C, 1]
        shift = shift[:, :, None]  # [B, C, 1]
        
        modulated = normalized * (1 + scale * spatial_weight * 0.1) + shift * spatial_weight * 0.1
        
        # 与原始特征混合（保持结构）
        # CHANGED: 将alpha从0.3降低到0.1，使风格调制成为更精细的微调，
        # 避免不稳定的风格特征过度破坏原始的几何特征。
        alpha = 0.1
        output = features * (1 - alpha) + modulated * alpha
        
        return output


class UnsupervisedPointCloudDiffusionModel(nn.Module):
    """修复的无监督点云Diffusion模型"""
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: List[int] = [128, 256, 512, 1024],
                 time_dim: int = 256,
                 style_dim: int = 256,
                 content_dims: List[int] = [64, 128, 256]):
        super().__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # 风格和内容编码器
        self.style_encoder = StyleEncoder(input_dim, style_dim=style_dim)
        self.content_encoder = ContentEncoder(input_dim, hidden_dims=content_dims)
        
        # 输入投影 - 包括内容特征
        content_dim = content_dims[-1]
        self.input_proj = nn.Conv1d(input_dim + content_dim, hidden_dims[0], 1)
        
        # 空间感知的风格调制
        self.style_modulation = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.style_modulation.append(
                SpatialAwareStyleModulation(hidden_dims[i], style_dim)
            )
        
        # 为中间块也添加风格调制
        self.middle_style_modulation = SpatialAwareStyleModulation(hidden_dims[-1], style_dim)
        
        # 编码器块
        self.encoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.encoder_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], time_dim)
            )
        
        # 中间块
        self.middle_block = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_dim)
        
        # 解码器块
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            in_channels = hidden_dims[-(i+1)] * 2  # skip connection
            out_channels = hidden_dims[-(i+2)]
            self.decoder_blocks.append(
                ResidualBlock(in_channels, out_channels, time_dim)
            )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dims[0], hidden_dims[0] // 2, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dims[0] // 2, input_dim, 1)
        )
        
        # 直接残差连接 - 保持输入结构
        self.input_skip = nn.Conv1d(input_dim, input_dim, 1)
        
        # 保存配置
        self.hidden_dims = hidden_dims
        self.style_dim = style_dim
        
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
            style_condition: Optional[torch.Tensor] = None,
            content_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 带噪声的点云 [B, N, 3]
            t: 时间步 [B]
            style_condition: 风格条件 [B, style_dim] (可选)
            content_condition: 内容条件 [B, C, N] (可选)
        Returns:
            预测的噪声 [B, N, 3]
        """
        B, N, _ = x.shape
        
        # 时间嵌入
        time_emb = self.time_embed(t)  # [B, time_dim]
        
        # 如果没有提供风格条件，从输入中提取
        if style_condition is None:
            style_condition = self.style_encoder(x)
        
        # 如果没有提供内容条件，从输入中提取
        if content_condition is None:
            content_condition = self.content_encoder(x)
        
        # 保存输入用于残差连接
        x_input = x.transpose(1, 2)  # [B, 3, N]
        
        # 准备输入 - 拼接内容特征
        x = torch.cat([x_input, content_condition], dim=1)  # [B, 3+C, N]
        h = self.input_proj(x)  # [B, hidden_dims[0], N]
        
        # 编码器
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, time_emb)
            # 轻量级风格调制 - 使用正确的索引
            if style_condition is not None:
                h = self.style_modulation[i+1](h, style_condition)
            skip_connections.append(h)
        
        # 中间块
        h = self.middle_block(h, time_emb)
        if style_condition is not None:
            h = self.middle_style_modulation(h, style_condition)
        
        # 解码器
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_emb)
        
        # 输出
        out = self.output_proj(h)  # [B, 3, N]
        
        # 强残差连接 - 确保保持结构
        out = out + self.input_skip(x_input) * 0.5
        
        out = out.transpose(1, 2)  # [B, N, 3]
        
        return out


class UnsupervisedDiffusionProcess:
    """无监督Diffusion过程 - 使用更温和的噪声调度"""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_schedule: str = "cosine", # 默认使用cosine
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02, # 在timesteps=1000时使用0.02
                 device: str = "cuda"):
        self.num_timesteps = num_timesteps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 创建beta调度
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps).to(self.device)
        else: # linear
            self.betas = self._linear_beta_schedule(num_timesteps, beta_start, beta_end).to(self.device)
        
        # 预计算alpha值
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        
        # 预计算采样所需的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def _linear_beta_schedule(self, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32)**2

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(x_start.shape[0], 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(x_start.shape[0], 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, 
                 style_condition: Optional[torch.Tensor] = None,
                 content_condition: Optional[torch.Tensor] = None,
                 clip_denoised: bool = True) -> torch.Tensor:
        """
        反向采样步骤 (DDPM)
        """
        betas_t = self.betas[t].view(x.shape[0], 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(x.shape[0], 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(x.shape[0], 1, 1)
        
        # 预测噪声
        predicted_noise = model(x, t, style_condition, content_condition)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if clip_denoised:
            model_mean = torch.clamp(model_mean, -1.5, 1.5)

        posterior_variance_t = self.posterior_variance[t].view(x.shape[0], 1, 1)
        
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...], 
            style_condition: Optional[torch.Tensor] = None,
            content_condition: Optional[torch.Tensor] = None,
            num_inference_steps: Optional[int] = 50) -> torch.Tensor:
        """
        生成采样 (使用DDIM加速)
        """
        device = next(model.parameters()).device
        img = torch.randn(shape, device=device)
        
        # 设置采样时间步
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for i in range(len(timesteps)):
            t = timesteps[i]
            prev_t = timesteps[i+1] if i < len(timesteps) - 1 else torch.tensor(-1, device=device)
            
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(img, batch_t, style_condition, content_condition)
            
            # DDIM 更新
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
            
            # 预测x0
            pred_x0 = (img - torch.sqrt(1. - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # 计算方向导数
            dir_xt = torch.sqrt(1. - alpha_t_prev) * predicted_noise
            
            # 更新x_t -> x_{t-1}
            img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return img

# 测试代码
if __name__ == "__main__":
    print("Testing Fixed Unsupervised Diffusion Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = UnsupervisedPointCloudDiffusionModel(
        input_dim=3,
        hidden_dims=[128, 256, 512, 1024],
        time_dim=256,
        style_dim=256
    ).to(device)
    
    # 测试数据
    batch_size = 2
    num_points = 2048
    x = torch.randn(batch_size, num_points, 3).to(device) * 0.5
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    # 提取风格和内容
    style = model.style_encoder(x)
    content = model.content_encoder(x)
    print(f"Style shape: {style.shape}")
    print(f"Content shape: {content.shape}")
    # 检查内容特征的空间方差，修复后应该大于0
    print(f"Content spatial variance: {content.var(dim=2).mean().item():.6f}")
    
    # 前向传播
    try:
        output = model(x, t)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        # 检查梯度
        loss = output.mean()
        loss.backward()
        print("✓ Gradient check passed!")
        print("✓ Model forward test passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()