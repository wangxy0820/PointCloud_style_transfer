import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.checkpoint import checkpoint


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    K近邻搜索
    Args:
        x: 输入点云特征 [B, C, N]
        k: 近邻数量
    Returns:
        近邻索引 [B, N, k]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [B, N, k]
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    构建图特征（完全修复版本）
    Args:
        x: 输入特征 [B, C, N]
        k: 近邻数量
        idx: 近邻索引
    Returns:
        图特征 [B, 2*C, N, k]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.contiguous()  # 确保连续性
    
    if idx is None:
        # 减少k值以节省内存
        k = min(k, 10)  # 限制最大k值
        idx = knn(x, k=k)
    
    device = x.device
    
    # 使用reshape代替view
    idx_base = torch.arange(0, batch_size, device=device).reshape(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.reshape(-1)
    
    _, num_dims, _ = x.size()
    
    # 确保x是连续的并转置
    x = x.transpose(2, 1).contiguous()  # [B, N, C]
    
    # 重塑并索引
    x_flat = x.reshape(batch_size * num_points, -1)
    feature = x_flat[idx, :]
    feature = feature.reshape(batch_size, num_points, k, num_dims)
    
    # 扩展x用于计算差异
    x_repeat = x.reshape(batch_size, num_points, 1, num_dims).expand(-1, -1, k, -1)
    
    # 构建特征
    feature = torch.cat((feature - x_repeat, x_repeat), dim=3)  # [B, N, k, 2*C]
    feature = feature.permute(0, 3, 1, 2).contiguous()  # [B, 2*C, N, k]
    
    return feature


class EdgeConv(nn.Module):
    """边卷积层（内存优化版）"""
    
    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super(EdgeConv, self).__init__()
        self.k = min(k, 10)  # 限制k值以节省内存
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取图特征
        x = get_graph_feature(x, k=self.k)  # [B, 2*C, N, k]
        
        # 卷积
        x = self.conv(x)  # [B, out_channels, N, k]
        
        # 最大池化
        x = x.max(dim=-1, keepdim=False)[0]  # [B, out_channels, N]
        
        return x


class PointNet2Backbone(nn.Module):
    """PointNet++骨干网络（内存优化版）"""
    
    def __init__(self, input_channels: int = 3, 
                 output_channels: List[int] = [64, 128, 256, 512],
                 k: int = 20):
        super(PointNet2Backbone, self).__init__()
        self.k = min(k, 10)  # 限制k值
        
        # 边卷积层
        self.edge_convs = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in output_channels:
            self.edge_convs.append(EdgeConv(in_channels, out_channels, self.k))
            in_channels = out_channels
        
        # 全局特征提取
        self.conv1 = nn.Conv1d(sum(output_channels), 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        
        # 输出层
        self.output_channels = output_channels[-1]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入点云 [B, 3, N]
        Returns:
            局部特征和全局特征
        """
        # 确保输入是连续的
        x = x.contiguous()
        
        # 边卷积提取局部特征
        features = []
        for i, edge_conv in enumerate(self.edge_convs):
            x = edge_conv(x)
            features.append(x)
            
            # 每两层清理一次缓存
            if i % 2 == 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 连接所有层的特征
        x = torch.cat(features, dim=1)  # [B, sum(output_channels), N]
        
        # 全局特征提取
        global_feature = F.relu(self.bn1(self.conv1(x)))  # [B, 1024, N]
        global_feature = F.adaptive_max_pool1d(global_feature, 1)  # [B, 1024, 1]
        
        # 使用squeeze代替view
        global_feature = global_feature.squeeze(-1)  # [B, 1024]
        
        return x, global_feature


class PointNet2Encoder(nn.Module):
    """PointNet++编码器"""
    
    def __init__(self, input_channels: int = 3,
                 feature_channels: List[int] = [64, 128, 256, 512],
                 output_dim: int = 512,
                 k: int = 20):
        super(PointNet2Encoder, self).__init__()
        
        self.backbone = PointNet2Backbone(input_channels, feature_channels, k)
        
        # 特征融合网络
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(sum(feature_channels), 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # 全局特征投影
        self.global_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码点云特征
        Args:
            x: 输入点云 [B, N, 3]
        Returns:
            局部特征 [B, output_dim, N] 和全局特征 [B, output_dim]
        """
        # 转换为 [B, 3, N] 格式
        x = x.transpose(2, 1).contiguous()
        
        # 提取特征
        local_features, global_feature = self.backbone(x)
        
        # 融合局部特征
        local_features = self.fusion_conv(local_features)  # [B, output_dim, N]
        
        # 投影全局特征
        global_feature = self.global_proj(global_feature)  # [B, output_dim]
        
        return local_features, global_feature


class PointNet2Decoder(nn.Module):
    """PointNet++解码器（动态版本）"""
    
    def __init__(self, input_dim: int = 512,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_channels: int = 3,
                 num_points: int = 8192):
        super(PointNet2Decoder, self).__init__()
        
        # 移除硬编码的点数
        self.input_dim = input_dim
        
        # 上采样网络
        self.upsample_layers = nn.ModuleList()
        in_dim = input_dim * 2  # 全局特征 + 局部特征
        
        for hidden_dim in hidden_dims:
            self.upsample_layers.append(nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ))
            in_dim = hidden_dim
        
        # 输出层
        self.output_conv = nn.Conv1d(in_dim, output_channels, 1)
        
    def forward(self, local_features: torch.Tensor, 
                global_feature: torch.Tensor) -> torch.Tensor:
        """
        解码点云
        Args:
            local_features: 局部特征 [B, input_dim, N]
            global_feature: 全局特征 [B, input_dim]
        Returns:
            重建的点云 [B, N, 3]
        """
        batch_size, _, num_points = local_features.shape
        
        # 动态扩展全局特征到匹配点数
        expanded_global = global_feature.unsqueeze(2).expand(-1, -1, num_points)  # [B, input_dim, N]
        
        # 融合局部和全局特征
        x = torch.cat([local_features, expanded_global], dim=1)  # [B, input_dim*2, N]
        
        # 上采样
        for layer in self.upsample_layers:
            x = layer(x)
        
        # 输出点云坐标
        output = self.output_conv(x)  # [B, 3, N]
        output = output.transpose(2, 1).contiguous()  # [B, N, 3]
        
        return output


class PointNet2AutoEncoder(nn.Module):
    """PointNet++自编码器"""
    
    def __init__(self, input_channels: int = 3,
                 feature_channels: List[int] = [64, 128, 256, 512],
                 latent_dim: int = 512,
                 num_points: int = 8192):
        super(PointNet2AutoEncoder, self).__init__()
        
        self.encoder = PointNet2Encoder(input_channels, feature_channels, latent_dim)
        self.decoder = PointNet2Decoder(latent_dim, [512, 256, 128], input_channels, num_points)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        自编码器前向传播
        Args:
            x: 输入点云 [B, N, 3]
        Returns:
            重建点云, 局部特征, 全局特征
        """
        local_features, global_feature = self.encoder(x)
        reconstructed = self.decoder(local_features, global_feature)
        
        return reconstructed, local_features, global_feature
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅编码
        Args:
            x: 输入点云 [B, N, 3]
        Returns:
            局部特征, 全局特征
        """
        return self.encoder(x)
    
    def decode(self, local_features: torch.Tensor, 
               global_feature: torch.Tensor) -> torch.Tensor:
        """
        仅解码
        Args:
            local_features: 局部特征
            global_feature: 全局特征
        Returns:
            重建点云
        """
        return self.decoder(local_features, global_feature)


def test_pointnet2():
    """测试PointNet++模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = PointNet2AutoEncoder(
        input_channels=3,
        feature_channels=[32, 64, 128, 256],  # 减小通道数
        latent_dim=256,  # 减小潜在维度
        num_points=2048  # 减小点数
    ).to(device)
    
    # 创建测试数据
    batch_size = 2
    num_points = 2048
    x = torch.randn(batch_size, num_points, 3).to(device)
    
    # 前向传播
    try:
        reconstructed, local_features, global_feature = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Local features shape: {local_features.shape}")
        print(f"Global features shape: {global_feature.shape}")
        print("✓ Forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试内存使用
    if torch.cuda.is_available():
        print(f"\nMemory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    test_pointnet2()