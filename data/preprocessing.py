import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import List, Tuple
import os


class ImprovedPointCloudPreprocessor:
    """改进的点云预处理器 - LiDAR感知版本"""
    
    def __init__(self, 
                 total_points: int = 120000,
                 chunk_size: int = 2048,
                 overlap_ratio: float = 0.3,
                 use_lidar_normalization: bool = True):
        self.total_points = total_points
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.overlap_size = int(chunk_size * overlap_ratio)
        self.stride = chunk_size - self.overlap_size
        self.use_lidar_normalization = use_lidar_normalization
    
    def normalize_point_cloud(self, points: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        标准化点云 - 支持LiDAR友好的标准化
        """
        if self.use_lidar_normalization:
            return self.normalize_point_cloud_lidar(points)
        else:
            return self.normalize_point_cloud_standard(points)
    
    def normalize_point_cloud_standard(self, points: np.ndarray) -> Tuple[np.ndarray, dict]:
        """标准的点云标准化（原方法）"""
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
        
        # 保存标准化参数
        norm_params = {
            'center': center,
            'scale': scale,
            'max_dist': max_dist,
            'method': 'standard'
        }
        
        return points_normalized, norm_params
    
    def normalize_point_cloud_lidar(self, points: np.ndarray) -> Tuple[np.ndarray, dict]:
        """LiDAR友好的标准化方法 - 保持扫描结构"""
        # 只在XY平面上进行中心化，保持Z轴（高度）信息
        xy_center = points[:, :2].mean(axis=0)
        points_normalized = points.copy()
        points_normalized[:, :2] -= xy_center
        
        # 计算数据范围但不进行激进的缩放
        xy_range = np.max(np.abs(points_normalized[:, :2]))
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        # 轻微的缩放，保持相对关系
        # 只有当范围过大时才缩放
        if xy_range > 100:  # 如果XY范围超过100米
            xy_scale = 50.0 / xy_range  # 缩放到100米范围内
            points_normalized[:, :2] *= xy_scale
        else:
            xy_scale = 1.0
        
        # Z轴单独处理，保持地面和高度信息
        z_center = (z_min + z_max) / 2
        z_range = z_max - z_min
        if z_range > 0:
            # 轻微标准化Z轴，但保持相对高度
            points_normalized[:, 2] = (points[:, 2] - z_center) / (z_range / 10)
        
        # 保存标准化参数
        norm_params = {
            'xy_center': xy_center,
            'xy_scale': xy_scale,
            'z_center': z_center,
            'z_range': z_range,
            'original_shape': points.shape,
            'method': 'lidar'
        }
        
        return points_normalized, norm_params
    
    def denormalize_point_cloud(self, points: np.ndarray, norm_params: dict) -> np.ndarray:
        """还原点云标准化"""
        if norm_params.get('method') == 'lidar':
            return self.denormalize_point_cloud_lidar(points, norm_params)
        else:
            # 标准方法
            return points / norm_params['scale'] + norm_params['center']
    
    def denormalize_point_cloud_lidar(self, points: np.ndarray, norm_params: dict) -> np.ndarray:
        """还原LiDAR点云标准化"""
        points_denorm = points.copy()
        
        # 还原Z轴
        if norm_params['z_range'] > 0:
            points_denorm[:, 2] = points[:, 2] * (norm_params['z_range'] / 10) + norm_params['z_center']
        
        # 还原XY平面
        points_denorm[:, :2] = points[:, :2] / norm_params['xy_scale'] + norm_params['xy_center']
        
        return points_denorm
    
    def create_overlapping_chunks(self, points: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """创建固定大小的重叠块 - 支持LiDAR感知的分块"""
        if self.use_lidar_normalization:
            # 使用LiDAR感知的分块方法
            return self.create_lidar_aware_chunks(points)
        else:
            # 使用原始的分块方法
            return self.create_standard_chunks(points)
    
    def create_standard_chunks(self, points: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """原始的分块方法"""
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
        
        # 如果点云太小，无法创建完整的chunk
        if len(chunks) == 0:
            # 通过重复采样创建至少一个chunk
            if n_points < self.chunk_size:
                indices = np.random.choice(n_points, self.chunk_size, replace=True)
                chunk_points = points[indices]
            else:
                chunk_points = points[:self.chunk_size]
            
            chunks.append((chunk_points, (0, min(n_points, self.chunk_size))))
        
        return chunks
    
    def create_lidar_aware_chunks(self, points: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """基于LiDAR扫描模式创建块"""
        n_points = len(points)
        chunks = []
        
        # 转换到球坐标系
        r = np.linalg.norm(points[:, :2], axis=1)  # XY平面的距离
        theta = np.arctan2(points[:, 1], points[:, 0])  # 水平角度 [-pi, pi]
        z = points[:, 2]  # 高度
        
        # 将角度归一化到 [0, 2*pi]
        theta_normalized = (theta + np.pi) % (2 * np.pi)
        
        # 按角度分区 - 将360度分成多个扇区
        n_angular_sectors = 16  # 每个扇区22.5度
        angular_width = 2 * np.pi / n_angular_sectors
        
        # 收集所有扇区的块
        all_sector_indices = []
        
        for sector_idx in range(n_angular_sectors):
            # 计算扇区的角度范围
            angle_start = sector_idx * angular_width
            angle_end = (sector_idx + 1) * angular_width
            
            # 找到属于这个扇区的点（考虑重叠）
            overlap_angle = angular_width * self.overlap_ratio
            expanded_start = angle_start - overlap_angle
            expanded_end = angle_end + overlap_angle
            
            # 处理角度环绕
            if expanded_start < 0:
                mask = (theta_normalized >= (expanded_start + 2*np.pi)) | (theta_normalized < expanded_end)
            elif expanded_end > 2*np.pi:
                mask = (theta_normalized >= expanded_start) | (theta_normalized < (expanded_end - 2*np.pi))
            else:
                mask = (theta_normalized >= expanded_start) & (theta_normalized < expanded_end)
            
            sector_indices = np.where(mask)[0]
            
            if len(sector_indices) >= self.chunk_size:
                # 在扇区内按高度排序
                z_values = z[sector_indices]
                sorted_idx = np.argsort(z_values)
                sorted_sector_indices = sector_indices[sorted_idx]
                
                # 创建固定大小的块
                for i in range(0, len(sorted_sector_indices) - self.chunk_size + 1, self.stride):
                    chunk_indices = sorted_sector_indices[i:i + self.chunk_size]
                    chunk_points = points[chunk_indices]
                    
                    # 记录原始索引范围
                    start_idx = chunk_indices.min()
                    end_idx = chunk_indices.max() + 1
                    
                    chunks.append((chunk_points, (start_idx, end_idx)))
        
        # 如果扇区方法产生的块太少，补充一些随机块
        if len(chunks) < 10:
            # 使用原始方法补充
            additional_chunks = self.create_standard_chunks(points)
            chunks.extend(additional_chunks[:max(10 - len(chunks), 5)])
        
        # 确保所有chunk大小正确
        for i, (chunk, _) in enumerate(chunks):
            if len(chunk) != self.chunk_size:
                raise ValueError(f"Chunk {i} has incorrect size: {len(chunk)} != {self.chunk_size}")
        
        print(f"Created {len(chunks)} LiDAR-aware chunks of size {self.chunk_size}")
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
        print(f"Normalization method: {sim_params.get('method', 'unknown')}")
        
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
            'num_real_chunks': len(real_chunks),
            'use_lidar_normalization': self.use_lidar_normalization
        }
        
        save_path = os.path.join(output_dir, f'{file_id}_preprocessed.pt')
        torch.save(data, save_path)
        
        return save_path


# 测试函数
if __name__ == "__main__":
    # 测试LiDAR感知的预处理器
    preprocessor = ImprovedPointCloudPreprocessor(
        total_points=120000,
        chunk_size=2048,
        overlap_ratio=0.3,
        use_lidar_normalization=True  # 启用LiDAR模式
    )
    
    # 创建测试数据 - 模拟LiDAR扫描
    n_points = 120000
    # 生成类似LiDAR的扫描模式
    angles = np.random.uniform(-np.pi, np.pi, n_points)
    distances = np.random.uniform(5, 50, n_points)  # 5-50米范围
    heights = np.random.uniform(-2, 10, n_points)   # -2到10米高度
    
    test_points = np.zeros((n_points, 3))
    test_points[:, 0] = distances * np.cos(angles)
    test_points[:, 1] = distances * np.sin(angles)
    test_points[:, 2] = heights
    
    # 测试标准化
    norm_points, norm_params = preprocessor.normalize_point_cloud(test_points)
    print(f"Original range: [{test_points.min():.2f}, {test_points.max():.2f}]")
    print(f"Normalized range: [{norm_points.min():.2f}, {norm_points.max():.2f}]")
    print(f"Normalization params: {norm_params}")
    
    # 测试分块
    chunks = preprocessor.create_overlapping_chunks(norm_points)
    print(f"Created {len(chunks)} chunks")
    
    # 验证所有块大小
    sizes = [len(chunk[0]) for chunk in chunks]
    print(f"Chunk sizes: min={min(sizes)}, max={max(sizes)}, unique={set(sizes)}")