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
    """风格编码器 - 提取域不变特征"""
    
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
    """内容编码器 - 保留几何结构"""
    
    def __init__(self, input_dim: int = 3, hidden_dims: List[int] = [64, 128, 256]):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_dim, h_dim, 1),
                nn.GroupNorm(8, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 3]
        Returns:
            content: [B, C, N]
        """
        x = x.transpose(1, 2)  # [B, 3, N]
        content = self.encoder(x)
        return content


class UnsupervisedPointCloudDiffusionModel(nn.Module):
    """无监督点云Diffusion模型 - 修复版"""
    
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
        
        # 风格调制层 - 确保输出维度正确
        self.style_modulation = nn.ModuleList()
        for i in range(len(hidden_dims)):
            # 创建调制层，输出 hidden_dims[i] * 2 (scale和shift)
            self.style_modulation.append(
                nn.Sequential(
                    nn.Linear(style_dim, hidden_dims[i] * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dims[i] * 2, hidden_dims[i] * 2)
                )
            )
        
        # 为中间块也添加风格调制
        self.middle_style_modulation = nn.Sequential(
            nn.Linear(style_dim, hidden_dims[-1] * 2),
            nn.SiLU(),
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1] * 2)
        )
        
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
        
        # 保存配置
        self.hidden_dims = hidden_dims
        self.style_dim = style_dim
        
        # 打印架构信息用于调试
        print("Fixed Model Architecture:")
        print(f"  Hidden dimensions: {hidden_dims}")
        print(f"  Style dimension: {style_dim}")
        print(f"  Style modulation layers: {len(self.style_modulation)}")
        for i, mod in enumerate(self.style_modulation):
            print(f"    Layer {i}: expects {hidden_dims[i]} channels")
    
    def style_modulate(self, features: torch.Tensor, style: torch.Tensor, 
                      layer_idx: int, is_middle: bool = False) -> torch.Tensor:
        """风格调制 - AdaIN风格"""
        B, C, N = features.shape
        
        # 获取scale和shift
        if is_middle:
            style_params = self.middle_style_modulation(style)  # [B, C*2]
        else:
            style_params = self.style_modulation[layer_idx](style)  # [B, C*2]
        
        # 确保维度正确
        expected_dim = self.hidden_dims[-1] if is_middle else self.hidden_dims[layer_idx]
        if C != expected_dim:
            raise ValueError(f"Feature dimension mismatch at layer {layer_idx}: "
                           f"expected {expected_dim}, got {C}")
        
        scale, shift = style_params.chunk(2, dim=1)  # 各[B, C]
        
        # 应用调制
        scale = scale[:, :, None]  # [B, C, 1]
        shift = shift[:, :, None]  # [B, C, 1]
        
        # 归一化特征
        mean = features.mean(dim=[2], keepdim=True)
        std = features.std(dim=[2], keepdim=True) + 1e-5
        features_norm = (features - mean) / std
        
        # 应用风格
        return features_norm * scale + shift
    
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
        
        # 准备输入 - 拼接内容特征
        x = x.transpose(1, 2)  # [B, 3, N]
        x = torch.cat([x, content_condition], dim=1)  # [B, 3+C, N]
        
        # 输入投影
        h = self.input_proj(x)  # [B, hidden_dims[0], N]
        
        # 编码器
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, time_emb)
            # 应用风格调制 - 使用正确的索引
            # 注意：编码器块i的输出维度是hidden_dims[i+1]
            h = self.style_modulate(h, style_condition, i+1)
            skip_connections.append(h)
        
        # 中间块
        h = self.middle_block(h, time_emb)
        h = self.style_modulate(h, style_condition, 0, is_middle=True)
        
        # 解码器
        for i, block in enumerate(self.decoder_blocks):
            # 跳跃连接
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_emb)
            # 注意：解码器不需要风格调制
        
        # 输出
        out = self.output_proj(h)  # [B, 3, N]
        out = out.transpose(1, 2)  # [B, N, 3]
        
        return out


class UnsupervisedDiffusionProcess:
    """无监督Diffusion过程"""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_schedule: str = "cosine",
                 device: str = "cuda"):
        self.num_timesteps = num_timesteps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 创建beta调度
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps).to(self.device)
        else:
            self.betas = self._linear_beta_schedule(num_timesteps).to(self.device)
        
        # 预计算alpha值
        self.alphas = (1 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        
        # 预计算采样所需的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)
    
    def _linear_beta_schedule(self, timesteps: int) -> torch.Tensor:
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        device = x_start.device
        t = t.to(device)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, 
                 style_condition: Optional[torch.Tensor] = None,
                 content_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        反向采样步骤
        """
        device = x.device
        t = t.to(device)
        
        # 预测噪声
        predicted_noise = model(x, t, style_condition, content_condition)
        
        # 获取系数并确保在正确的设备上
        betas_t = self.betas[t].to(device)[:, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].to(device)[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)[:, None, None]
        
        mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)
        
        if t[0] > 0:
            noise = torch.randn_like(x)
            # 计算后验方差
            alphas_cumprod_t = self.alphas_cumprod[t].to(device)[:, None, None]
            alphas_cumprod_prev_t = self.alphas_cumprod_prev[t].to(device)[:, None, None]
            posterior_variance_t = betas_t * (1 - alphas_cumprod_prev_t) / (1 - alphas_cumprod_t)
            return mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...], 
               style_condition: Optional[torch.Tensor] = None,
               content_condition: Optional[torch.Tensor] = None,
               num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        生成采样
        """
        device = next(model.parameters()).device
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
        
        # DDIM采样
        if num_inference_steps < self.num_timesteps:
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, 
                num_inference_steps, dtype=torch.long, device=device
            )
            
            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                t_prev = timesteps[i + 1]
                
                batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
                predicted_noise = model(x, batch_t, style_condition, content_condition)
                
                # DDIM更新
                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = self.alphas_cumprod[t_prev]
                
                x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                x0_pred = torch.clamp(x0_pred, -2, 2)
                
                x = torch.sqrt(alpha_t_prev) * x0_pred + \
                    torch.sqrt(1 - alpha_t_prev) * predicted_noise
                
                x = torch.clamp(x, -3, 3)
        else:
            # 完整DDPM采样
            for t in reversed(range(self.num_timesteps)):
                batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
                x = self.p_sample(model, x, batch_t, style_condition, content_condition)
        
        return torch.clamp(x, -1.5, 1.5)


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
    x = torch.randn(batch_size, num_points, 3).to(device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    # 提取风格和内容
    style = model.style_encoder(x)
    content = model.content_encoder(x)
    print(f"Style shape: {style.shape}")
    print(f"Content shape: {content.shape}")
    
    # 前向传播
    try:
        output = model(x, t)
        print(f"Output shape: {output.shape}")
        print("✓ Model test passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()