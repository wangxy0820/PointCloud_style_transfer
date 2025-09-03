# models/pointnet2_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """计算两组点云之间的平方欧氏距离。"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """根据索引从点云中提取点。"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    idx_clamped = torch.clamp(idx, 0, points.shape[1] - 1)
    new_points = points[batch_indices, idx_clamped, :]
    return new_points

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """最远点采样。"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """球查询。"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, 
                 in_channel: int, mlp: List[int], group_all: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = xyz.shape
        
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3).to(xyz.device)
            if points is not None:
                grouped_points = torch.cat([xyz.view(B, 1, N, 3), points.view(B, 1, N, -1)], dim=-1)
            else:
                grouped_points = xyz.view(B, 1, N, 3)
            new_points = self.apply_mlp(grouped_points)
            
            new_points = new_points.squeeze(-1)  # [B, C, 1, N] -> [B, C, N]
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)
            
            if points is not None:
                grouped_points = index_points(points, group_idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
            new_points = self.apply_mlp(new_points)
            
        return new_xyz, new_points
        
    def apply_mlp(self, points):
        # points: [B, S, nsample, C] -> [B, C, S, nsample]
        points = points.permute(0, 3, 1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            points = F.relu(bn(conv(points)))
        # 取最大值池化: [B, C, S, nsample] -> [B, C, S]
        return torch.max(points, 3)[0]

class PointNet2Encoder(nn.Module):
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        
        self.sa1 = SetAbstraction(512, 0.2, 32, in_channel=0, mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, in_channel=128, mlp=[128, 128, 256]) 
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=None, 
                                 in_channel=256, mlp=[256, 512, feature_dim], group_all=True)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        B, N, C = xyz.shape
        points = None  # 初始时没有特征
        
        l1_xyz, l1_points = self.sa1(xyz, points)  # l1_points: [B, 128, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points.permute(0, 2, 1))  # [B, 256, 128]  
        _, global_feature = self.sa3(l2_xyz, l2_points.permute(0, 2, 1))  # [B, feature_dim, 1]
        
        return global_feature.view(B, -1)  # [B, feature_dim]