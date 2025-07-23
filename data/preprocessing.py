import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import List, Tuple
import os


class ImprovedPointCloudPreprocessor:
    """改进的点云预处理器 - 确保chunk大小一致"""
    
    def __init__(self, 
                 total_points: int = 120000,
                 chunk_size: int = 2048,
                 overlap_ratio: float = 0.3):
        self.total_points = total_points
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.overlap_size = int(chunk_size * overlap_ratio)
        self.stride = chunk_size - self.overlap_size
    
    def normalize_point_cloud(self, points: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        标准化点云到单位球内
        """
        # 计算中心
        center = points.mean(axis=0)
        points_centered = points - center
        
        # 计算缩放因子，确保点云在单位球内
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        
        if max_dist > 0:
            scale = 1.0 / max_dist  # 归一化到[-1, 1]
            points_normalized = points_centered * scale
        else:
            scale = 1.0
            points_normalized = points_centered
        
        # 验证归一化
        assert np.max(np.abs(points_normalized)) <= 1.0 + 1e-6, "Normalization failed"
        
        # 保存标准化参数
        norm_params = {
            'center': center,
            'scale': scale,
            'max_dist': max_dist
        }
        
        return points_normalized, norm_params
    
    def denormalize_point_cloud(self, points: np.ndarray, norm_params: dict) -> np.ndarray:
        """还原点云标准化"""
        return points / norm_params['scale'] + norm_params['center']
    
    def create_overlapping_chunks(self, points: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        创建固定大小的重叠块
        """
        n_points = len(points)
        chunks = []
        
        # 方法1: 使用滑动窗口确保每个chunk大小完全一致
        if n_points >= self.chunk_size:
            # 计算可以创建多少个完整的chunk
            num_full_chunks = (n_points - self.chunk_size) // self.stride + 1
            
            for i in range(num_full_chunks):
                start_idx = i * self.stride
                end_idx = start_idx + self.chunk_size
                
                # 确保不超出范围
                if end_idx <= n_points:
                    chunk_points = points[start_idx:end_idx]
                    assert len(chunk_points) == self.chunk_size, f"Chunk size mismatch: {len(chunk_points)} != {self.chunk_size}"
                    chunks.append((chunk_points, (start_idx, end_idx)))
            
            # 处理剩余的点（如果有）
            if num_full_chunks * self.stride < n_points - self.chunk_size:
                # 从末尾创建一个chunk
                start_idx = n_points - self.chunk_size
                end_idx = n_points
                chunk_points = points[start_idx:end_idx]
                assert len(chunk_points) == self.chunk_size, f"Last chunk size mismatch: {len(chunk_points)} != {self.chunk_size}"
                chunks.append((chunk_points, (start_idx, end_idx)))
        
        # 如果点云太小，无法创建完整的chunk
        if len(chunks) == 0:
            # 通过重复采样创建至少一个chunk
            if n_points < self.chunk_size:
                indices = np.random.choice(n_points, self.chunk_size, replace=True)
                chunk_points = points[indices]
            else:
                chunk_points = points[:self.chunk_size]
            
            chunks.append((chunk_points, (0, min(n_points, self.chunk_size))))
        
        # 方法2: 基于空间聚类的方法（可选，用于更好的空间覆盖）
        if len(chunks) < 10:  # 如果chunk太少，使用聚类方法补充
            n_clusters = max(10 - len(chunks), min(20, n_points // self.chunk_size))
            
            if n_clusters > 0 and n_points > n_clusters:
                # K-means聚类
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(points)
                centers = kmeans.cluster_centers_
                
                for i in range(n_clusters):
                    # 找到离每个聚类中心最近的chunk_size个点
                    distances = np.linalg.norm(points - centers[i], axis=1)
                    nearest_indices = np.argsort(distances)[:self.chunk_size]
                    
                    # 如果不够，重复采样
                    if len(nearest_indices) < self.chunk_size:
                        additional_indices = np.random.choice(
                            nearest_indices, 
                            self.chunk_size - len(nearest_indices), 
                            replace=True
                        )
                        nearest_indices = np.concatenate([nearest_indices, additional_indices])
                    
                    chunk_points = points[nearest_indices]
                    assert len(chunk_points) == self.chunk_size, f"Cluster chunk size mismatch: {len(chunk_points)} != {self.chunk_size}"
                    
                    start_idx = min(nearest_indices)
                    end_idx = max(nearest_indices) + 1
                    chunks.append((chunk_points, (start_idx, end_idx)))
        
        # 最终验证：确保所有chunk大小一致
        for i, (chunk, _) in enumerate(chunks):
            if len(chunk) != self.chunk_size:
                raise ValueError(f"Chunk {i} has incorrect size: {len(chunk)} != {self.chunk_size}")
        
        print(f"Created {len(chunks)} chunks of size {self.chunk_size}")
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
        
        # 验证标准化结果
        print(f"Sim points range after norm: [{sim_norm.min():.3f}, {sim_norm.max():.3f}]")
        print(f"Real points range after norm: [{real_norm.min():.3f}, {real_norm.max():.3f}]")
        
        # 创建块
        sim_chunks = self.create_overlapping_chunks(sim_norm)
        real_chunks = self.create_overlapping_chunks(real_norm)
        
        # 确保所有chunk大小正确
        for i, (chunk, _) in enumerate(sim_chunks):
            assert len(chunk) == self.chunk_size, f"Sim chunk {i} size error: {len(chunk)}"
        for i, (chunk, _) in enumerate(real_chunks):
            assert len(chunk) == self.chunk_size, f"Real chunk {i} size error: {len(chunk)}"
        
        # 保存数据
        data = {
            'sim_chunks': sim_chunks,
            'real_chunks': real_chunks,
            'sim_norm_params': sim_params,
            'real_norm_params': real_params,
            'total_points': self.total_points,
            'chunk_size': self.chunk_size,  # 明确保存chunk_size
            'overlap_ratio': self.overlap_ratio,
            'num_sim_chunks': len(sim_chunks),
            'num_real_chunks': len(real_chunks)
        }
        
        save_path = os.path.join(output_dir, f'{file_id}_preprocessed.pt')
        torch.save(data, save_path)
        
        return save_path


# 测试函数
if __name__ == "__main__":
    # 测试预处理器
    preprocessor = ImprovedPointCloudPreprocessor(
        total_points=120000,
        chunk_size=2048,
        overlap_ratio=0.3
    )
    
    # 创建测试数据
    test_points = np.random.randn(120000, 3) * 100  # 大范围的点
    
    # 测试标准化
    norm_points, norm_params = preprocessor.normalize_point_cloud(test_points)
    print(f"Original range: [{test_points.min():.2f}, {test_points.max():.2f}]")
    print(f"Normalized range: [{norm_points.min():.2f}, {norm_points.max():.2f}]")
    
    # 测试分块
    chunks = preprocessor.create_overlapping_chunks(norm_points)
    print(f"Created {len(chunks)} chunks")
    
    # 验证所有块大小
    sizes = [len(chunk[0]) for chunk in chunks]
    print(f"Chunk sizes: min={min(sizes)}, max={max(sizes)}, unique={set(sizes)}")