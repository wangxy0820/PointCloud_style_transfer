import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import List, Tuple
import os


class ImprovedPointCloudPreprocessor:
    """改进的点云预处理器"""
    
    def __init__(self, 
                 total_points: int = 120000,
                 chunk_size: int = 2048,
                 overlap_ratio: float = 0.3):
        self.total_points = total_points
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.overlap_size = int(chunk_size * overlap_ratio)
    
    def normalize_point_cloud(self, points: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        标准化点云并返回标准化参数
        """
        # 计算中心和缩放
        center = points.mean(axis=0)
        points_centered = points - center
        scale = np.max(np.linalg.norm(points_centered, axis=1))
        
        if scale > 0:
            points_normalized = points_centered / scale
        else:
            points_normalized = points_centered
        
        # 保存标准化参数用于还原
        norm_params = {
            'center': center,
            'scale': scale
        }
        
        return points_normalized, norm_params
    
    def denormalize_point_cloud(self, points: np.ndarray, norm_params: dict) -> np.ndarray:
        """还原点云标准化"""
        return points * norm_params['scale'] + norm_params['center']
    
    def create_overlapping_chunks(self, points: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        创建有重叠的块
        Returns:
            List of (chunk_points, (start_idx, end_idx))
        """
        n_points = len(points)
        
        # 使用空间聚类创建块
        n_clusters = max(1, n_points // (self.chunk_size - self.overlap_size))
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
        centers = kmeans.cluster_centers_
        
        chunks = []
        used_indices = set()
        
        for i in range(n_clusters):
            # 获取属于当前聚类的点
            cluster_mask = labels == i
            cluster_indices = np.where(cluster_mask)[0]
            
            # 计算到聚类中心的距离
            distances = np.linalg.norm(points - centers[i], axis=1)
            
            # 选择最近的chunk_size个点（包括一些其他聚类的点作为重叠）
            nearest_indices = np.argsort(distances)[:self.chunk_size]
            
            # 确保有足够的重叠
            overlap_candidates = []
            for j in range(n_clusters):
                if i != j:
                    # 从相邻聚类中选择一些点
                    other_cluster_mask = labels == j
                    other_indices = np.where(other_cluster_mask)[0]
                    if len(other_indices) > 0:
                        # 选择离当前聚类中心较近的点
                        other_distances = distances[other_indices]
                        nearest_other = other_indices[np.argsort(other_distances)[:self.overlap_size // n_clusters]]
                        overlap_candidates.extend(nearest_other)
            
            # 组合核心点和重叠点
            if overlap_candidates:
                all_indices = np.unique(np.concatenate([cluster_indices, overlap_candidates]))
                # 根据距离排序并选择最近的chunk_size个
                all_indices = all_indices[np.argsort(distances[all_indices])[:self.chunk_size]]
            else:
                all_indices = nearest_indices
            
            chunk_points = points[all_indices]
            
            # 记录块的位置信息
            start_idx = min(all_indices)
            end_idx = max(all_indices) + 1
            
            chunks.append((chunk_points, (start_idx, end_idx)))
            used_indices.update(all_indices)
        
        # 确保所有点都被包含
        unused_indices = set(range(n_points)) - used_indices
        if unused_indices:
            # 将未使用的点添加到最近的块中
            unused_indices = np.array(list(unused_indices))
            unused_points = points[unused_indices]
            
            # 找到每个未使用点最近的聚类中心
            distances_to_centers = np.array([
                np.linalg.norm(unused_points - center, axis=1) 
                for center in centers
            ]).T
            
            nearest_clusters = np.argmin(distances_to_centers, axis=1)
            
            # 将点添加到对应的块
            for i, idx in enumerate(unused_indices):
                cluster_id = nearest_clusters[i]
                chunk_points, (start_idx, end_idx) = chunks[cluster_id]
                
                # 更新块
                new_points = np.vstack([chunk_points, points[idx:idx+1]])
                new_end = max(end_idx, idx + 1)
                chunks[cluster_id] = (new_points, (start_idx, new_end))
        
        return chunks
    
    def save_preprocessed_data(self, 
                              sim_points: np.ndarray,
                              real_points: np.ndarray,
                              output_dir: str,
                              file_id: str):
        """保存预处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 标准化
        sim_norm, sim_params = self.normalize_point_cloud(sim_points)
        real_norm, real_params = self.normalize_point_cloud(real_points)
        
        # 创建块
        sim_chunks = self.create_overlapping_chunks(sim_norm)
        real_chunks = self.create_overlapping_chunks(real_norm)
        
        # 保存数据
        data = {
            'sim_chunks': sim_chunks,
            'real_chunks': real_chunks,
            'sim_norm_params': sim_params,
            'real_norm_params': real_params,
            'total_points': self.total_points,
            'chunk_size': self.chunk_size,
            'overlap_ratio': self.overlap_ratio
        }
        
        save_path = os.path.join(output_dir, f'{file_id}_preprocessed.pt')
        torch.save(data, save_path)
        
        return save_path
