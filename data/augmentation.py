import torch
import numpy as np
from typing import Tuple

class PointCloudAugmentation:
    """点云数据增强 - 保持空间顺序"""
    
    def __init__(self,
                 rotation_range: float = 0.05,
                 jitter_std: float = 0.005,
                 scale_range: Tuple[float, float] = (0.98, 1.02),
                 shuffle_points: bool = False): # 默认禁用shuffle
        self.rotation_range = rotation_range
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.shuffle_points = shuffle_points
    
    def random_rotation(self, points: torch.Tensor) -> torch.Tensor:
        """随机Z轴旋转"""
        if self.rotation_range <= 0: return points
        angles = torch.rand(points.shape[0], device=points.device) * 2 * self.rotation_range - self.rotation_range
        cos, sin = torch.cos(angles), torch.sin(angles)
        rotation_matrices = torch.zeros(points.shape[0], 3, 3, device=points.device)
        rotation_matrices[:, 0, 0] = cos
        rotation_matrices[:, 0, 1] = -sin
        rotation_matrices[:, 1, 0] = sin
        rotation_matrices[:, 1, 1] = cos
        rotation_matrices[:, 2, 2] = 1
        return torch.bmm(points, rotation_matrices)

    def random_jitter(self, points: torch.Tensor) -> torch.Tensor:
        """随机抖动"""
        if self.jitter_std <= 0: return points
        noise = torch.randn_like(points) * self.jitter_std
        return points + noise
    
    def random_scale(self, points: torch.Tensor) -> torch.Tensor:
        """随机缩放"""
        if self.scale_range[0] == 1.0 and self.scale_range[1] == 1.0: return points
        scales = torch.rand(points.shape[0], 1, 1, device=points.device) * \
                 (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        return points * scales
    
    def shuffle_points_order(self, points: torch.Tensor) -> torch.Tensor:
        """打乱点的顺序"""
        if not self.shuffle_points: return points
        for b in range(points.shape[0]):
            perm = torch.randperm(points.shape[1], device=points.device)
            points[b] = points[b, perm]
        return points
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        is_single = points.dim() == 2
        if is_single:
            points = points.unsqueeze(0)
        
        augmented = points
        augmented = self.random_rotation(augmented)
        augmented = self.random_jitter(augmented)
        augmented = self.random_scale(augmented)
        
        # 关键：不再默认执行shuffle，除非显式开启
        augmented = self.shuffle_points_order(augmented)
        
        if is_single:
            augmented = augmented.squeeze(0)
        
        return augmented

def create_lidar_augmentation(config):
    """根据配置创建LiDAR友好的数据增强"""
    return PointCloudAugmentation(
        rotation_range=config.augmentation_rotation_range,
        jitter_std=config.augmentation_jitter_std,
        scale_range=config.augmentation_scale_range,
        shuffle_points=False  # 明确禁用Shuffle
    )
