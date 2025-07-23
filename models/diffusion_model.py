import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
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


class CrossAttention(nn.Module):
    """交叉注意力模块用于风格转换"""
    
    def __init__(self, dim: int, context_dim: int = 1024, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] - 内容特征
            context: [B, M, context_dim] - 风格特征
        """
        B, N, C = x.shape
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 多头注意力
        q = q.reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        k = k.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)
        v = v.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)
        
        # 计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        return self.to_out(out)


class PointCloudDiffusionModel(nn.Module):
    """点云Diffusion模型用于风格转换"""
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: List[int] = [128, 256, 512, 1024],
                 time_dim: int = 256,
                 context_dim: int = 1024,
                 num_heads: int = 8):
        super().__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # 输入投影
        self.input_proj = nn.Conv1d(input_dim, hidden_dims[0], 1)
        
        # 编码器
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.encoder_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], time_dim)
            )
            self.encoder_attns.append(
                CrossAttention(hidden_dims[i+1], context_dim, num_heads)
            )
        
        # 中间块
        self.middle_block = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_dim)
        self.middle_attn = CrossAttention(hidden_dims[-1], context_dim, num_heads)
        
        # 解码器 - 正确的维度计算
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        
        # 解码器维度：
        # hidden_dims = [128, 256, 512, 1024]
        # 解码器0: 1024(来自middle) + 512(skip) = 1536 -> 512
        # 解码器1: 512(来自decoder0) + 256(skip) = 768 -> 256  
        # 解码器2: 256(来自decoder1) + 128(skip) = 384 -> 128
        
        for i in range(len(hidden_dims) - 1):
            in_channels = hidden_dims[- (i + 1)] * 2  # 输入是前一层输出通道数的两倍
            out_channels = hidden_dims[- (i + 2)]     # 输出通道数逐步减少
            self.decoder_blocks.append(
                ResidualBlock(in_channels, out_channels, time_dim)
            )
            self.decoder_attns.append(
                CrossAttention(out_channels, context_dim, num_heads)
            )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dims[0], hidden_dims[0] // 2, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dims[0] // 2, input_dim, 1)
        )
        
        # 打印架构信息
        print("Model Architecture:")
        print(f"  Input projection: {input_dim} -> {hidden_dims[0]}")
        for i in range(len(hidden_dims) - 1):
            print(f"  Encoder {i}: {hidden_dims[i]} -> {hidden_dims[i+1]}")
        print(f"  Middle block: {hidden_dims[-1]} -> {hidden_dims[-1]}")
        for i in range(len(hidden_dims) - 1):
            in_channels = hidden_dims[- (i + 1)] * 2
            out_channels = hidden_dims[- (i + 2)]
            print(f"  Decoder {i}: {in_channels} -> {out_channels}")
        print(f"  Output projection: {hidden_dims[0]} -> {input_dim}")
        
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor,
                style_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 带噪声的点云 [B, N, 3]
            t: 时间步 [B]
            style_features: 风格特征 [B, M, context_dim]
        Returns:
            预测的噪声 [B, N, 3]
        """
        # 调试：检查输入形状
        # print(f"Forward pass - Input x: {x.shape}")
        
        # 准备输入
        x = x.transpose(1, 2)  # [B, 3, N]
        time_emb = self.time_embed(t)  # [B, time_dim]
        
        # 输入投影
        h = self.input_proj(x)  # [B, 128, N]
        
        # 编码器
        skip_connections = []
        for i, (block, attn) in enumerate(zip(self.encoder_blocks, self.encoder_attns)):
            h = block(h, time_emb)
            # 交叉注意力用于风格转换
            h_transpose = h.transpose(1, 2)  # [B, N, C]
            h_attn = attn(h_transpose, style_features)
            h = h + h_attn.transpose(1, 2)
            skip_connections.append(h)
            # print(f"Encoder {i}: {h.shape}")
        
        # 中间块
        h = self.middle_block(h, time_emb)
        h_transpose = h.transpose(1, 2)
        h_attn = self.middle_attn(h_transpose, style_features)
        h = h + h_attn.transpose(1, 2)
        # print(f"Middle block: {h.shape}")
        
        # 解码器
        for i, (block, attn) in enumerate(zip(self.decoder_blocks, self.decoder_attns)):
            # 跳跃连接
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=1)
            # print(f"Decoder {i} - after concat: {h.shape}, block expects: {block.conv1.in_channels}")
            
            h = block(h, time_emb)
            # 交叉注意力
            h_transpose = h.transpose(1, 2)
            h_attn = attn(h_transpose, style_features)
            h = h + h_attn.transpose(1, 2)
        
        # 输出
        out = self.output_proj(h)  # [B, 3, N]
        out = out.transpose(1, 2)  # [B, N, 3]
        
        return out


class DiffusionProcess:
    """Diffusion过程管理"""
    
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
        
        # 预计算alpha值 - 确保所有张量都在正确的设备上
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
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向扩散过程
        Args:
            x_start: 原始数据 [B, N, 3]
            t: 时间步 [B]
            noise: 噪声 [B, N, 3]
        Returns:
            带噪声的数据
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 确保索引在正确的设备上
        device = x_start.device
        t = t.to(device)
        
        # 获取对应时间步的系数
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        
        # 添加维度以便广播
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, 
                 style_features: torch.Tensor) -> torch.Tensor:
        """
        反向采样步骤
        Args:
            model: 噪声预测模型
            x: 当前带噪声的数据 [B, N, 3]
            t: 时间步 [B]
            style_features: 风格特征
        Returns:
            去噪后的数据
        """
        device = x.device
        t = t.to(device)
        
        # 预测噪声
        predicted_noise = model(x, t, style_features)
        
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
               style_features: torch.Tensor) -> torch.Tensor:
        """
        完整的采样过程
        Args:
            model: 噪声预测模型
            shape: 输出形状 (B, N, 3)
            style_features: 风格特征
        Returns:
            生成的点云
        """
        device = next(model.parameters()).device
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for t in reversed(range(self.num_timesteps)):
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, batch_t, style_features)
        
        return x


# 测试代码
if __name__ == "__main__":
    print("Testing Diffusion Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = PointCloudDiffusionModel(
        input_dim=3,
        hidden_dims=[128, 256, 512, 1024],
        time_dim=256,
        context_dim=1024,
        num_heads=8
    ).to(device)
    
    # 测试数据
    batch_size = 2
    num_points = 2048
    x = torch.randn(batch_size, num_points, 3).to(device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    style_features = torch.randn(batch_size, 1, 1024).to(device)
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  t: {t.shape}")
    print(f"  style_features: {style_features.shape}")
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        output = model(x, t, style_features)
        print(f"\nOutput shape: {output.shape}")
        print("✓ Model test passed!")