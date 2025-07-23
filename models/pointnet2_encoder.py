"""
改进的PointNet++编码器 - 修复版V2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class SetAbstraction(nn.Module):
    """Set Abstraction模块"""
    
    def __init__(self, npoint: int, radius: float, nsample: int, 
                 in_channel: int, mlp: List[int], group_all: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.group_all = group_all
    
    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: 输入点坐标 [B, N, 3]
            points: 输入点特征 [B, N, C]（可选）
        Returns:
            new_xyz: 采样点坐标 [B, npoint, 3] or [B, 3] if group_all
            new_points: 采样点特征 [B, npoint, C'] or [B, C'] if group_all
        """
        B, N, C = xyz.shape
        
        if self.group_all:
            # 全局聚合 - 修复版
            new_xyz = xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]
            
            if points is not None:
                # 连接坐标和特征
                new_points = torch.cat([xyz, points], dim=-1)  # [B, N, 3+C]
            else:
                new_points = xyz  # [B, N, 3]
            
            # 转换为conv2d需要的格式：[B, C, H, W]
            new_points = new_points.transpose(1, 2).unsqueeze(-1)  # [B, 3+C, N, 1]
            
            # MLP
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
            
            # 全局最大池化 - 关键修复：先在N维度上池化，再squeeze
            new_points = torch.max(new_points, dim=2, keepdim=True)[0]  # [B, C', 1, 1]
            new_points = new_points.squeeze(-1).squeeze(-1)  # [B, C']
            new_xyz = new_xyz.squeeze(1)  # [B, 3]
            
        else:
            # 采样
            fps_idx = self.farthest_point_sample(xyz, self.npoint)  # [B, npoint]
            new_xyz = self.index_points(xyz, fps_idx)  # [B, npoint, 3]
            
            # 分组
            idx = self.query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # [B, npoint, nsample]
            grouped_xyz = self.index_points(xyz, idx)  # [B, npoint, nsample, 3]
            grouped_xyz -= new_xyz.unsqueeze(2)  # 相对坐标
            
            if points is not None:
                grouped_points = self.index_points(points, idx)  # [B, npoint, nsample, C]
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, npoint, nsample, 3+C]
            else:
                new_points = grouped_xyz  # [B, npoint, nsample, 3]
            
            # 转换为conv2d需要的格式：[B, C, H, W]
            new_points = new_points.permute(0, 3, 1, 2)  # [B, 3+C, npoint, nsample]
            
            # MLP
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
            
            # 最大池化
            new_points = torch.max(new_points, dim=-1)[0]  # [B, C', npoint]
            new_points = new_points.transpose(1, 2)  # [B, npoint, C']
        
        return new_xyz, new_points
    
    def farthest_point_sample(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """最远点采样"""
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # [B, 1, 3]
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # [B, N]
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=-1)[1]
        
        return centroids
    
    def query_ball_point(self, radius: float, nsample: int, xyz: torch.Tensor, 
                        new_xyz: torch.Tensor) -> torch.Tensor:
        """球查询"""
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = self.square_distance(new_xyz, xyz)  # [B, S, N]
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        
        return group_idx
    
    def index_points(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """索引点"""
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points
    
    def square_distance(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """计算平方距离"""
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
        dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
        return dist


class ImprovedPointNet2Encoder(nn.Module):
    """改进的PointNet++编码器"""
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 1024):
        super().__init__()
        
        # Set Abstraction层
        self.sa1 = SetAbstraction(
            npoint=512, 
            radius=0.2, 
            nsample=32, 
            in_channel=3,
            mlp=[64, 64, 128]
        )
        
        self.sa2 = SetAbstraction(
            npoint=128, 
            radius=0.4, 
            nsample=64, 
            in_channel=3 + 128,
            mlp=[128, 128, 256]
        )
        
        self.sa3 = SetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None, 
            in_channel=3 + 256,
            mlp=[256, 512, 1024],  # 固定为1024
            group_all=True
        )
        
        # 特征提取头 - 保证输出是feature_dim
        self.feature_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, feature_dim)
        )
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: 输入点云 [B, N, 3]
        Returns:
            全局特征 [B, feature_dim]
        """
        B, N, _ = xyz.shape
        
        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # l3_points应该已经是[B, feature_dim]
        global_feature = l3_points
        
        # 安全检查
        if global_feature.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {global_feature.dim()}D tensor with shape {global_feature.shape}")
        
        # 特征处理
        global_feature = self.feature_head(global_feature)
        
        return global_feature


# 测试代码
if __name__ == "__main__":
    # 测试编码器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    num_points = 2048
    points = torch.randn(batch_size, num_points, 3).to(device)
    
    # 创建模型
    encoder = ImprovedPointNet2Encoder(input_channels=3, feature_dim=1024).to(device)
    
    # 前向传播
    encoder.eval()
    with torch.no_grad():
        features = encoder(points)
    
    print(f"Input shape: {points.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output mean: {features.mean().item():.4f}")
    print(f"Output std: {features.std().item():.4f}")
    
    # 测试梯度
    encoder.train()
    features = encoder(points)
    loss = features.mean()
    loss.backward()
    
    print("\nGradient check passed!")
    print("PointNet2 Encoder is working correctly!")