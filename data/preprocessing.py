import numpy as np
import torch
from typing import Tuple, Dict
import os
from sklearn.neighbors import NearestNeighbors

class PointCloudPreprocessor:
    """分层点云预处理器 - 修复归一化与性能问题"""
    
    def __init__(self, 
                 total_points: int = 120000,
                 global_points: int = 30000):
        self.total_points = total_points
        self.global_points = global_points
        
        print(f"初始化高效预处理器:")
        print(f"  - 总点数: {self.total_points}")
        print(f"  - 全局点数: {self.global_points}")
        print(f"  - 下采样算法: Voxel Grid Downsampling")

    def normalize_point_cloud(self, points: np.ndarray, target_range: float = 1.8) -> Tuple[np.ndarray, dict]:
        """等比例归一化，保持几何形状"""
        center = points.mean(axis=0)
        points_centered = points - center
        max_abs_value = np.max(np.abs(points_centered))
        
        if max_abs_value < 1e-6: # 避免除以零
            scale = 1.0
        else:
            scale = target_range / max_abs_value
        
        points_normalized = points_centered * scale
        
        norm_params = {
            'center': center, 'scale': scale, 
            'method': 'isotropic', 'target_range': target_range
        }
        return points_normalized, norm_params

    def denormalize_point_cloud(self, points: np.ndarray, norm_params: dict) -> np.ndarray:
        """反归一化"""
        return (points / norm_params['scale']) + norm_params['center']

    # --- NEW: 高效的体素化网格下采样实现 ---
    def _voxel_grid_downsample_numpy(self, points: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用体素化网格进行高效下采样, 返回点和索引。
        """
        n_points = points.shape[0]
        if n_points <= target_size:
            return points, np.arange(n_points)

        # 估算合适的体素大小
        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)
        xyz_range = xyz_max - xyz_min
        
        # 避免范围为0的情况
        xyz_range[xyz_range < 1e-6] = 1.0 

        # 估算体素大小以获得接近目标数量的点
        # 乘以一个系数（如1.2）来轻微增大体素，确保初始采样点数不会远超目标
        voxel_size = (xyz_range.prod() / target_size)**(1/3) * 1.2 
        if voxel_size < 1e-6:
            voxel_size = 1e-3 # 设置一个最小体素大小

        # 将点分配到体素中
        voxel_indices = np.floor((points - xyz_min) / voxel_size).astype(int)
        
        # 使用字典来存储每个体素中的点索引
        voxel_dict = {}
        for i in range(len(voxel_indices)):
            key = tuple(voxel_indices[i])
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)
            
        # 从每个体素中选择一个代表点（最接近体素中心）
        selected_indices = []
        for key, indices in voxel_dict.items():
            voxel_points = points[indices]
            voxel_center = xyz_min + (np.array(key) + 0.5) * voxel_size
            center_distances = np.linalg.norm(voxel_points - voxel_center, axis=1)
            representative_index = indices[np.argmin(center_distances)]
            selected_indices.append(representative_index)
        
        # 如果采样点数不足，随机补充
        current_size = len(selected_indices)
        if current_size < target_size:
            remaining_needed = target_size - current_size
            all_indices = set(np.arange(n_points))
            selected_set = set(selected_indices)
            pool = list(all_indices - selected_set)
            
            if len(pool) > 0:
                additional_indices = np.random.choice(pool, min(remaining_needed, len(pool)), replace=False)
                selected_indices.extend(additional_indices)

        # 如果采样点数过多，随机丢弃
        elif current_size > target_size:
            selected_indices = np.random.choice(selected_indices, target_size, replace=False)
            
        final_indices = np.array(selected_indices, dtype=int)
        return points[final_indices], final_indices
    # ---------------------------------------------

    def consistent_downsample(self, points: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        MODIFIED: 调用新的高效下采样方法
        """
        return self._voxel_grid_downsample_numpy(points, target_size)
    
    # ... consistent_upsample 方法是高效的，无需修改 ...
    def consistent_upsample(self, coarse_points: np.ndarray, original_points: np.ndarray, 
                           coarse_indices: np.ndarray) -> np.ndarray:
        N = len(original_points); M = len(coarse_points)
        result = np.zeros((N, 3), dtype=np.float32)
        result[coarse_indices] = coarse_points
        unknown_mask = np.ones(N, dtype=bool); unknown_mask[coarse_indices] = False
        unknown_indices = np.where(unknown_mask)[0]
        if len(unknown_indices) > 0:
            k = min(3, M); nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(original_points[coarse_indices])
            distances, neighbors = nbrs.kneighbors(original_points[unknown_indices])
            weights = 1.0 / (distances + 1e-8); weights = weights / weights.sum(axis=1, keepdims=True)
            interpolated = np.sum(coarse_points[neighbors] * weights[:, :, np.newaxis], axis=1)
            result[unknown_indices] = interpolated
        return result
    
    def create_hierarchical_data(self, points: np.ndarray) -> Dict:
        """创建分层数据结构"""
        points_norm, norm_params = self.normalize_point_cloud(points)
        global_points, global_indices = self.consistent_downsample(points_norm, self.global_points)
        return {
            'full_points': points_norm, 'global_points': global_points,
            'global_indices': global_indices, 'norm_params': norm_params
        }
    
    def save_hierarchical_data(self, sim_points: np.ndarray, real_points: np.ndarray,
                              output_dir: str, file_id: str):
        """保存分层预处理数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # --- MODIFIED: 重采样步骤现在也使用高效的体素化下采样 ---
        if len(sim_points) != self.total_points:
            print(f"Resampling sim_points from {len(sim_points)} to {self.total_points}")
            if len(sim_points) > self.total_points:
                sim_points, _ = self._voxel_grid_downsample_numpy(sim_points, self.total_points)
            else:
                indices = np.random.choice(len(sim_points), self.total_points, replace=True)
                sim_points = sim_points[indices]
        
        if len(real_points) != self.total_points:
            print(f"Resampling real_points from {len(real_points)} to {self.total_points}")
            if len(real_points) > self.total_points:
                real_points, _ = self._voxel_grid_downsample_numpy(real_points, self.total_points)
            else:
                indices = np.random.choice(len(real_points), self.total_points, replace=True)
                real_points = real_points[indices]
        # ----------------------------------------------------

        sim_data = self.create_hierarchical_data(sim_points)
        real_data = self.create_hierarchical_data(real_points)
        
        data = {
            'sim_full': sim_data['full_points'], 'sim_global': sim_data['global_points'],
            'sim_global_indices': sim_data['global_indices'], 'sim_norm_params': sim_data['norm_params'],
            'real_full': real_data['full_points'], 'real_global': real_data['global_points'],
            'real_global_indices': real_data['global_indices'], 'real_norm_params': real_data['norm_params'],
            'total_points': self.total_points, 'global_points': self.global_points
        }
        
        save_path = os.path.join(output_dir, f'{file_id}_hierarchical.pt')
        torch.save(data, save_path)
        print(f"Successfully saved hierarchical data to {save_path}")
        return save_path