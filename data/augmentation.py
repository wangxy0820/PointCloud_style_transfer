"""
数据增强模块 - LiDAR友好版本
"""

import torch
import numpy as np
from typing import Tuple, Optional


class PointCloudAugmentation:
    """点云数据增强 - 针对LiDAR调整"""
    
    def __init__(self,
                 rotation_range: float = 0.05,  # 减小旋转范围
                 jitter_std: float = 0.005,     # 减小抖动
                 scale_range: Tuple[float, float] = (0.98, 1.02),  # 减小缩放范围
                 dropout_ratio: float = 0.0,
                 shuffle_points: bool = True,
                 preserve_scan_pattern: bool = True):  # 新增：是否保持扫描模式
        """
        Args:
            rotation_range: 旋转角度范围（弧度）- LiDAR数据应该使用较小值
            jitter_std: 抖动噪声标准差 - 过大会破坏扫描线
            scale_range: 缩放范围 - LiDAR数据应该保持原始尺度
            dropout_ratio: 随机丢弃点的比例
            shuffle_points: 是否打乱点的顺序
            preserve_scan_pattern: 是否尽量保持LiDAR扫描模式
        """
        self.rotation_range = rotation_range
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.dropout_ratio = dropout_ratio
        self.shuffle_points = shuffle_points
        self.preserve_scan_pattern = preserve_scan_pattern
    
    def random_rotation(self, points: torch.Tensor) -> torch.Tensor:
        """随机旋转（仅绕Z轴） - LiDAR友好版本"""
        if self.rotation_range <= 0:
            return points
        
        batch_size = points.shape[0]
        device = points.device
        
        # 随机旋转角度 - 对于LiDAR使用更小的范围
        angles = torch.rand(batch_size, device=device) * 2 * self.rotation_range - self.rotation_range
        
        # 构建旋转矩阵（仅Z轴旋转，保持垂直结构）
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
        """随机抖动 - LiDAR友好版本"""
        if self.jitter_std <= 0:
            return points
        
        # 对于LiDAR，不同方向使用不同的噪声强度
        if self.preserve_scan_pattern:
            noise = torch.randn_like(points)
            # XY平面使用完整噪声
            noise[:, :, :2] *= self.jitter_std
            # Z方向使用更小的噪声，保持垂直结构
            noise[:, :, 2] *= self.jitter_std * 0.5
        else:
            noise = torch.randn_like(points) * self.jitter_std
        
        return points + noise
    
    def random_scale(self, points: torch.Tensor) -> torch.Tensor:
        """随机缩放 - LiDAR友好版本"""
        if self.scale_range[0] == 1.0 and self.scale_range[1] == 1.0:
            return points
        
        batch_size = points.shape[0]
        device = points.device
        
        if self.preserve_scan_pattern:
            # 只在XY平面缩放，保持Z轴不变
            scales_xy = torch.rand(batch_size, 1, 1, device=device) * \
                       (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            
            scaled_points = points.clone()
            scaled_points[:, :, :2] *= scales_xy
            return scaled_points
        else:
            # 统一缩放
            scales = torch.rand(batch_size, 1, 1, device=device) * \
                    (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            return points * scales
    
    def random_dropout(self, points: torch.Tensor) -> torch.Tensor:
        """随机丢弃点 - LiDAR友好版本"""
        if self.dropout_ratio <= 0:
            return points
        
        batch_size, num_points, _ = points.shape
        
        # 随机选择保留的点
        keep_ratio = 1 - self.dropout_ratio
        keep_num = int(num_points * keep_ratio)
        
        augmented_points = []
        for b in range(batch_size):
            if self.preserve_scan_pattern:
                # 按扫描线丢弃，而不是完全随机
                # 将点按Z轴排序，模拟扫描线
                z_values = points[b, :, 2]
                sorted_indices = torch.argsort(z_values)
                
                # 随机选择要保留的扫描线段
                segment_size = num_points // 20  # 假设20条扫描线
                keep_segments = torch.randperm(20)[:int(20 * keep_ratio)]
                
                indices = []
                for seg in keep_segments:
                    start_idx = seg * segment_size
                    end_idx = min((seg + 1) * segment_size, num_points)
                    indices.extend(sorted_indices[start_idx:end_idx].tolist())
                
                indices = torch.tensor(indices[:keep_num], device=points.device)
            else:
                # 完全随机丢弃
                indices = torch.randperm(num_points, device=points.device)[:keep_num]
            
            kept_points = points[b, indices]
            
            # 填充到原始大小（通过重复）
            if keep_num < num_points:
                repeat_indices = torch.randint(0, keep_num, (num_points - keep_num,), device=points.device)
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
            perm = torch.randperm(num_points, device=points.device)
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


# 创建LiDAR特定的数据增强
def create_lidar_augmentation(config):
    """根据配置创建LiDAR友好的数据增强"""
    return PointCloudAugmentation(
        rotation_range=config.augmentation_rotation_range,
        jitter_std=config.augmentation_jitter_std,
        scale_range=config.augmentation_scale_range,
        dropout_ratio=0.0,  # LiDAR数据通常不使用dropout
        shuffle_points=True,
        preserve_scan_pattern=True
    )