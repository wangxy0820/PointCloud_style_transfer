# data/preprocessing.py

import numpy as np
import torch
from typing import Tuple, Dict
import os
from sklearn.neighbors import NearestNeighbors

class PointCloudPreprocessor:
    """分层点云预处理器"""
    
    def __init__(self, 
                 total_points: int = 120000,
                 global_points: int = 30000):
        self.total_points = total_points
        self.global_points = global_points
    
    def normalize_point_cloud(self, points: np.ndarray) -> Tuple[np.ndarray, dict]:
        """归一化"""
        center = points.mean(axis=0)
        points_centered = points - center
        
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        
        if max_dist > 0:
            scale = 1.0 / max_dist
        else:
            scale = 1.0
        
        points_normalized = points_centered * scale
        
        norm_params = {
            'center': center,
            'scale': scale,
            'method': 'isotropic'
        }
        
        return points_normalized, norm_params
    
    def denormalize_point_cloud(self, points: np.ndarray, norm_params: dict) -> np.ndarray:
        """还原点云标准化"""
        points_denorm = points / norm_params['scale']
        points_denorm = points_denorm + norm_params['center']
        return points_denorm
    
    def voxel_downsample(self, points: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """体素下采样
        返回:
            downsampled_points: 下采样后的点
            indices: 被选中点的原始索引
        """
        n_points = len(points)
        #print(f"    [Voxel DEBUG] Starting downsample from {n_points} to {target_size} points.")
        
        if n_points <= target_size:
            #print("    [Voxel DEBUG] Point count is already less than or equal to target. Returning original.")
            return points, np.arange(n_points)
        
        pts_min = points.min(axis=0)
        pts_max = points.max(axis=0)
        pts_range = pts_max - pts_min
        
        # 估算体素大小
        voxel_size = (pts_range.prod() / target_size) ** (1/3) * 1.2
        #print(f"    [Voxel DEBUG] Estimated voxel size: {voxel_size:.4f}")
        
        # 体素化
        #print("    [Voxel DEBUG] Building voxel dictionary...")
        voxel_dict = {}
        for i, pt in enumerate(points):
            voxel_key = tuple((pt / voxel_size).astype(int))
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = []
            voxel_dict[voxel_key].append(i)
        #print(f"    [Voxel DEBUG] Voxel dictionary built with {len(voxel_dict)} unique voxels.")
        
        # 从每个体素选择代表点
        selected_indices = []
        for voxel_indices in voxel_dict.values():
            voxel_points = points[voxel_indices]
            center = voxel_points.mean(axis=0)
            distances = np.linalg.norm(voxel_points - center, axis=1)
            selected_idx = voxel_indices[np.argmin(distances)]
            selected_indices.append(selected_idx)
        
        selected_indices = np.array(selected_indices)
        #print(f"    [Voxel DEBUG] Selected {len(selected_indices)} points from voxels.")
        
        # 如果点数不够，使用FPS补充
        if len(selected_indices) < target_size:
            remaining_needed = target_size - len(selected_indices)
            all_indices = set(range(n_points))
            # 获取所有未被选择的点的索引
            available_indices = list(all_indices - set(selected_indices))
            
            if available_indices and remaining_needed > 0:
                #print(f"    [Voxel DEBUG] Points are not enough. Supplementing {remaining_needed} points using FAST random sampling.")
                
                # 从未被选择的点中，快速、随机地选择所需数量的点
                num_to_sample = min(remaining_needed, len(available_indices))
                extra_indices = np.random.choice(available_indices, num_to_sample, replace=False)
                
                # 将随机选出的点与之前体素中心选出的点合并
                selected_indices = np.concatenate([selected_indices, extra_indices])
        
        #rint(f"    [Voxel DEBUG] Total selected points before final adjustment: {len(selected_indices)}.")
        # 如果还是太多，随机采样到目标数量
        if len(selected_indices) > target_size:
            #print("    [Voxel DEBUG] Points are too many. Randomly sampling down...")
            selected_indices = np.random.choice(selected_indices, target_size, replace=False)
        
        #print(f"    [Voxel DEBUG] Downsample finished. Final point count: {len(selected_indices)}.")
        return points[selected_indices], selected_indices
    
    def create_hierarchical_data(self, points: np.ndarray) -> Dict:
        """创建分层数据
        返回:
            包含完整点云和下采样点云的字典
        """
        # 归一化
        points_norm, norm_params = self.normalize_point_cloud(points)
        
        # 创建全局下采样
        global_points, global_indices = self.voxel_downsample(points_norm, self.global_points)
        
        return {
            'full_points': points_norm,  # [120000, 3]
            'global_points': global_points,  # [30000, 3]
            'global_indices': global_indices,  # 索引映射
            'norm_params': norm_params
        }
    
    def save_hierarchical_data(self, sim_points: np.ndarray, real_points: np.ndarray,
                              output_dir: str, file_id: str):
        """保存分层预处理数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 确保点数正确
        if len(sim_points) != self.total_points:
            # 重采样到目标点数
            if len(sim_points) > self.total_points:
                indices = np.random.choice(len(sim_points), self.total_points, replace=False)
                sim_points = sim_points[indices]
            else:
                indices = np.random.choice(len(sim_points), self.total_points, replace=True)
                sim_points = sim_points[indices]
        
        if len(real_points) != self.total_points:
            if len(real_points) > self.total_points:
                indices = np.random.choice(len(real_points), self.total_points, replace=False)
                real_points = real_points[indices]
            else:
                indices = np.random.choice(len(real_points), self.total_points, replace=True)
                real_points = real_points[indices]
        
        # 创建分层数据
        sim_data = self.create_hierarchical_data(sim_points)
        real_data = self.create_hierarchical_data(real_points)
        
        # 保存
        data = {
            'sim_full': sim_data['full_points'],
            'sim_global': sim_data['global_points'],
            'sim_global_indices': sim_data['global_indices'],
            'sim_norm_params': sim_data['norm_params'],
            'real_full': real_data['full_points'],
            'real_global': real_data['global_points'],
            'real_global_indices': real_data['global_indices'],
            'real_norm_params': real_data['norm_params'],
            'total_points': self.total_points,
            'global_points': self.global_points
        }
        
        save_path = os.path.join(output_dir, f'{file_id}_hierarchical.pt')
        torch.save(data, save_path)
        
        print(f"Saved hierarchical data: full={self.total_points}, global={self.global_points}")
        return save_path
    
    # 兼容旧接口（但不使用）
    def create_overlapping_chunks(self, points: np.ndarray):
        """弃用 - 仅为兼容性保留"""
        raise NotImplementedError("Chunks are deprecated. Use create_hierarchical_data instead.")