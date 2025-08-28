import numpy as np
import torch
from typing import List, Tuple
import os

class PointCloudPreprocessor:
    """
    最终修复的点云预处理器
    - 实现了正确的、保持长宽比的(各向同性)归一化方法。
    - 保证了数据在进入模型前，其几何结构是完整且未被扭曲的。
    """
    
    def __init__(self, 
                 total_points: int = 120000,
                 chunk_size: int = 4096,
                 overlap_ratio: float = 0.3):
        self.total_points = total_points
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.overlap_size = int(chunk_size * overlap_ratio)
        self.stride = chunk_size - self.overlap_size
    
    def normalize_point_cloud(self, points: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        正确的、保持长宽比的(各向同性)归一化方法。
        将点云等比例缩放到一个单位球内。
        """
        # 1. 计算中心点并移至原点
        center = points.mean(axis=0)
        points_centered = points - center
        
        # 2. 找到离原点最远的点的距离
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        
        # 3. 用这个最大距离进行等比例缩放
        if max_dist > 0:
            scale = 1.0 / max_dist
        else:
            scale = 1.0
        
        points_normalized = points_centered * scale
        
        # 验证范围：所有点到原点的距离都应小于等于1
        actual_max_dist = np.max(np.linalg.norm(points_normalized, axis=1))
        assert actual_max_dist <= 1.001, \
            f"Isotropic normalization failed: max distance is {actual_max_dist:.4f}"

        norm_params = {
            'center': center,
            'scale': scale, # 现在scale是一个单一值
            'method': 'isotropic'
        }
        
        return points_normalized, norm_params
    
    def denormalize_point_cloud(self, points: np.ndarray, norm_params: dict) -> np.ndarray:
        """还原点云标准化"""
        # 仅支持新的、正确的反归一化
        points_denorm = points / norm_params['scale']
        points_denorm = points_denorm + norm_params['center']
        return points_denorm

    def create_overlapping_chunks(self, points: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """创建重叠块 (逻辑保持不变，依然是鲁棒的滑动窗口)"""
        n_points = len(points)
        chunks = []
        
        if n_points < self.chunk_size:
            indices = np.random.choice(n_points, self.chunk_size, replace=True)
            chunk_points = points[indices]
            chunks.append((chunk_points, (0, n_points)))
            return chunks

        for i in range(0, n_points - self.chunk_size + 1, self.stride):
            start_idx = i
            end_idx = start_idx + self.chunk_size
            chunk_points = points[start_idx:end_idx]
            chunks.append((chunk_points, (start_idx, end_idx)))
        
        if not chunks or (len(chunks) > 0 and chunks[-1][1][1] < n_points):
             start_idx = n_points - self.chunk_size
             end_idx = n_points
             chunk_points = points[start_idx:end_idx]
             chunks.append((chunk_points, (start_idx, end_idx)))
            
        return chunks

    def save_preprocessed_data(self, sim_points: np.ndarray, real_points: np.ndarray,
                              output_dir: str, file_id: str):
        os.makedirs(output_dir, exist_ok=True)
        
        sim_norm, sim_params = self.normalize_point_cloud(sim_points)
        real_norm, real_params = self.normalize_point_cloud(real_points)
        
        sim_chunks = self.create_overlapping_chunks(sim_norm)
        real_chunks = self.create_overlapping_chunks(real_norm)
        
        for i, (chunk, _) in enumerate(sim_chunks):
            assert len(chunk) == self.chunk_size, f"Sim chunk {i} size error: {len(chunk)}"
        for i, (chunk, _) in enumerate(real_chunks):
            assert len(chunk) == self.chunk_size, f"Real chunk {i} size error: {len(chunk)}"
        
        data = {
            'sim_chunks': sim_chunks,
            'real_chunks': real_chunks,
            'sim_norm_params': sim_params,
            'real_norm_params': real_params,
            'chunk_size': self.chunk_size,
        }
        
        save_path = os.path.join(output_dir, f'{file_id}_preprocessed.pt')
        torch.save(data, save_path)
        return save_path
