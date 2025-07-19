import numpy as np
import os
import random
from typing import List, Tuple, Optional
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch


class PointCloudPreprocessor:
    """点云数据预处理器"""
    
    def __init__(self, chunk_size: int = 8192, overlap_ratio: float = 0.1):
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.scaler = StandardScaler()
        
    def normalize_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        标准化点云数据
        Args:
            points: 点云数据 [N, 3]
        Returns:
            标准化后的点云 [N, 3]
        """
        # 中心化
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # 缩放到单位球
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale
            
        return points
    
    def random_chunk(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """
        随机采样固定数量的点
        Args:
            points: 输入点云 [N, 3]
            num_points: 采样点数
        Returns:
            采样后的点云 [num_points, 3]
        """
        if len(points) >= num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
        else:
            # 如果点数不足，进行上采样
            indices = np.random.choice(len(points), num_points, replace=True)
        
        return points[indices]
    
    def spatial_chunk(self, points: np.ndarray, num_chunks: int) -> List[np.ndarray]:
        """
        基于空间位置进行分块
        Args:
            points: 输入点云 [N, 3]
            num_chunks: 分块数量
        Returns:
            分块后的点云列表
        """
        # 使用K-means聚类进行空间分块
        kmeans = KMeans(n_clusters=num_chunks, random_state=42)
        labels = kmeans.fit_predict(points)
        
        chunks = []
        for i in range(num_chunks):
            chunk_points = points[labels == i]
            if len(chunk_points) > 0:
                # 确保每个块有足够的点
                if len(chunk_points) < self.chunk_size:
                    # 重复采样到目标大小
                    indices = np.random.choice(len(chunk_points), self.chunk_size, replace=True)
                    chunk_points = chunk_points[indices]
                else:
                    # 随机采样到目标大小
                    chunk_points = self.random_chunk(chunk_points, self.chunk_size)
                
                chunks.append(self.normalize_point_cloud(chunk_points))
        
        return chunks
    
    def sliding_window_chunk(self, points: np.ndarray) -> List[np.ndarray]:
        """
        滑动窗口分块策略
        Args:
            points: 输入点云 [N, 3]
        Returns:
            分块后的点云列表
        """
        # 按Z轴排序
        sorted_indices = np.argsort(points[:, 2])
        sorted_points = points[sorted_indices]
        
        chunks = []
        step_size = int(self.chunk_size * (1 - self.overlap_ratio))
        
        for start_idx in range(0, len(sorted_points) - self.chunk_size + 1, step_size):
            end_idx = start_idx + self.chunk_size
            chunk = sorted_points[start_idx:end_idx]
            chunks.append(self.normalize_point_cloud(chunk))
            
        return chunks
    
    def augment_point_cloud(self, points: np.ndarray, 
                          rotation_range: float = 0.1,
                          jitter_std: float = 0.01,
                          scaling_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        点云数据增强
        Args:
            points: 输入点云 [N, 3]
            rotation_range: 旋转角度范围（弧度）
            jitter_std: 抖动标准差
            scaling_range: 缩放范围
        Returns:
            增强后的点云
        """
        augmented_points = points.copy()
        
        # 随机旋转
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            cos_angle, sin_angle = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1]
            ])
            augmented_points = augmented_points @ rotation_matrix.T
        
        # 添加噪声
        if jitter_std > 0:
            noise = np.random.normal(0, jitter_std, augmented_points.shape)
            augmented_points += noise
        
        # 随机缩放
        if scaling_range[0] != 1.0 or scaling_range[1] != 1.0:
            scale = np.random.uniform(scaling_range[0], scaling_range[1])
            augmented_points *= scale
            
        return augmented_points
    
    def process_file(self, file_path: str, output_dir: str, 
                    domain: str, file_id: str,
                    chunk_method: str = "spatial") -> List[str]:
        """
        处理单个点云文件
        Args:
            file_path: 输入文件路径
            output_dir: 输出目录
            domain: 域标签 ("sim" or "real")
            file_id: 文件ID
            chunk_method: 分块方法 ("spatial", "random", "sliding")
        Returns:
            生成的文件路径列表
        """
        # 加载点云数据
        points = np.load(file_path)
        if points.shape[1] > 3:
            points = points[:, :3]  # 只保留xyz坐标
        
        # 标准化
        points = self.normalize_point_cloud(points)
        
        # 分块
        if chunk_method == "spatial":
            chunks = self.spatial_chunk(points, num_chunks=15)
        elif chunk_method == "sliding":
            chunks = self.sliding_window_chunk(points)
        else:  # random
            chunks = []
            for i in range(15):  # 生成15个随机块
                chunk = self.random_chunk(points, self.chunk_size)
                chunks.append(self.normalize_point_cloud(chunk))
        
        # 保存分块
        output_paths = []
        domain_dir = os.path.join(output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        for i, chunk in enumerate(chunks):
            chunk_file = f"{file_id}_chunk_{i:03d}.npy"
            chunk_path = os.path.join(domain_dir, chunk_file)
            np.save(chunk_path, chunk.astype(np.float32))
            output_paths.append(chunk_path)
            
        return output_paths
    
    def create_dataset_split(self, data_dir: str, 
                           val_split: float = 0.1, 
                           test_split: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
        """
        创建数据集划分
        Args:
            data_dir: 数据目录
            val_split: 验证集比例
            test_split: 测试集比例
        Returns:
            训练集、验证集、测试集文件列表
        """
        all_files = []
        for domain in ["sim", "real"]:
            domain_dir = os.path.join(data_dir, domain)
            if os.path.exists(domain_dir):
                files = glob.glob(os.path.join(domain_dir, "*.npy"))
                all_files.extend([(f, domain) for f in files])
        
        # 随机打乱
        random.shuffle(all_files)
        
        # 划分数据集
        n_total = len(all_files)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_test - n_val
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        return train_files, val_files, test_files


def preprocess_dataset(sim_dir: str, real_dir: str, output_dir: str,
                      chunk_size: int = 8192, chunk_method: str = "spatial"):
    """
    预处理整个数据集
    Args:
        sim_dir: 仿真数据目录
        real_dir: 真实数据目录
        output_dir: 输出目录
        chunk_size: 分块大小
        chunk_method: 分块方法
    """
    preprocessor = PointCloudPreprocessor(chunk_size=chunk_size)
    
    # 处理仿真数据
    print("Processing simulation data...")
    sim_files = glob.glob(os.path.join(sim_dir, "*.npy"))
    for i, file_path in enumerate(sim_files):
        file_id = f"sim_{i:04d}"
        preprocessor.process_file(file_path, output_dir, "sim", file_id, chunk_method)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sim_files)} simulation files")
    
    # 处理真实数据
    print("Processing real world data...")
    real_files = glob.glob(os.path.join(real_dir, "*.npy"))
    for i, file_path in enumerate(real_files):
        file_id = f"real_{i:04d}"
        preprocessor.process_file(file_path, output_dir, "real", file_id, chunk_method)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(real_files)} real world files")
    
    # 创建数据集划分
    print("Creating dataset splits...")
    train_files, val_files, test_files = preprocessor.create_dataset_split(output_dir)
    
    # 保存数据集划分信息
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    import pickle
    with open(os.path.join(output_dir, "dataset_splits.pkl"), "wb") as f:
        pickle.dump(splits, f)
    
    print(f"Dataset preprocessing completed!")
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess point cloud dataset")
    parser.add_argument("--sim_dir", required=True, help="Simulation data directory")
    parser.add_argument("--real_dir", required=True, help="Real world data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Chunk size")
    parser.add_argument("--chunk_method", choices=["spatial", "random", "sliding"], 
                       default="spatial", help="Chunking method")
    
    args = parser.parse_args()
    
    preprocess_dataset(args.sim_dir, args.real_dir, args.output_dir, 
                      args.chunk_size, args.chunk_method)