"""
数据增强模块
"""

import torch
import numpy as np
from typing import Tuple, Optional


class PointCloudAugmentation:
    """点云数据增强"""
    
    def __init__(self,
                 rotation_range: float = 0.1,
                 jitter_std: float = 0.01,
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 dropout_ratio: float = 0.0,
                 shuffle_points: bool = True):
        """
        Args:
            rotation_range: 旋转角度范围（弧度）
            jitter_std: 抖动噪声标准差
            scale_range: 缩放范围
            dropout_ratio: 随机丢弃点的比例
            shuffle_points: 是否打乱点的顺序
        """
        self.rotation_range = rotation_range
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.dropout_ratio = dropout_ratio
        self.shuffle_points = shuffle_points
    
    def random_rotation(self, points: torch.Tensor) -> torch.Tensor:
        """随机旋转（仅绕Z轴）"""
        if self.rotation_range <= 0:
            return points
        
        batch_size = points.shape[0]
        device = points.device
        
        # 随机旋转角度
        angles = torch.rand(batch_size, device=device) * 2 * self.rotation_range - self.rotation_range
        
        # 构建旋转矩阵
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        rotation_matrices = torch.zeros(batch_size, 3, 3, device=device)
        rotation_matrices[:, 0, 0] = cos_angles
        rotation_matrices[:, 0, 1] = -sin_angles
        rotation_matrices[:, 1, 0] = sin_angles
        rotation_matrices[:, 1, 1] = cos_angles
        rotation_matrices[:, 2, 2] = 1
        
        # 应用旋转
        rotated_points = torch.bmm(points, rotation_matrices.transpose(1, 2))
        
        return rotated_points
    
    def random_jitter(self, points: torch.Tensor) -> torch.Tensor:
        """随机抖动"""
        if self.jitter_std <= 0:
            return points
        
        noise = torch.randn_like(points) * self.jitter_std
        return points + noise
    
    def random_scale(self, points: torch.Tensor) -> torch.Tensor:
        """随机缩放"""
        if self.scale_range[0] == 1.0 and self.scale_range[1] == 1.0:
            return points
        
        batch_size = points.shape[0]
        device = points.device
        
        # 随机缩放因子
        scales = torch.rand(batch_size, 1, 1, device=device) * \
                (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        
        return points * scales
    
    def random_dropout(self, points: torch.Tensor) -> torch.Tensor:
        """随机丢弃点"""
        if self.dropout_ratio <= 0:
            return points
        
        batch_size, num_points, _ = points.shape
        
        # 随机选择保留的点
        keep_ratio = 1 - self.dropout_ratio
        keep_num = int(num_points * keep_ratio)
        
        augmented_points = []
        for b in range(batch_size):
            # 随机选择索引
            indices = torch.randperm(num_points)[:keep_num]
            kept_points = points[b, indices]
            
            # 填充到原始大小（通过重复）
            if keep_num < num_points:
                repeat_indices = torch.randint(0, keep_num, (num_points - keep_num,))
                repeated_points = kept_points[repeat_indices]
                kept_points = torch.cat([kept_points, repeated_points], dim=0)
            
            augmented_points.append(kept_points)
        
        return torch.stack(augmented_points)
    
    def shuffle_points_order(self, points: torch.Tensor) -> torch.Tensor:
        """打乱点的顺序"""
        if not self.shuffle_points:
            return points
        
        batch_size, num_points, _ = points.shape
        
        # 为每个批次生成随机排列
        shuffled_points = []
        for b in range(batch_size):
            perm = torch.randperm(num_points)
            shuffled_points.append(points[b, perm])
        
        return torch.stack(shuffled_points)
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        应用所有增强
        Args:
            points: 输入点云 [B, N, 3] 或 [N, 3]
        Returns:
            增强后的点云
        """
        # 处理单个点云的情况
        if points.dim() == 2:
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 应用各种增强
        augmented = points
        augmented = self.random_rotation(augmented)
        augmented = self.random_jitter(augmented)
        augmented = self.random_scale(augmented)
        augmented = self.random_dropout(augmented)
        augmented = self.shuffle_points_order(augmented)
        
        if squeeze_output:
            augmented = augmented.squeeze(0)
        
        return augmented
