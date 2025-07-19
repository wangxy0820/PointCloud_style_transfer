import torch
import numpy as np
import os
import pickle
import random
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from .preprocess import PointCloudPreprocessor


class PointCloudDataset(Dataset):
    """点云数据集类"""
    
    def __init__(self, data_files: List[Tuple[str, str]], 
                 augment: bool = False,
                 augment_params: Optional[Dict] = None):
        """
        初始化数据集
        Args:
            data_files: 文件路径和域标签的列表 [(path, domain), ...]
            augment: 是否进行数据增强
            augment_params: 数据增强参数
        """
        self.data_files = data_files
        self.augment = augment
        self.augment_params = augment_params or {}
        
        if self.augment:
            self.preprocessor = PointCloudPreprocessor()
        
        # 分离不同域的数据
        self.sim_files = [(path, domain) for path, domain in data_files if domain == "sim"]
        self.real_files = [(path, domain) for path, domain in data_files if domain == "real"]
        
        print(f"Dataset initialized with {len(self.sim_files)} sim files and {len(self.real_files)} real files")
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        Args:
            idx: 样本索引
        Returns:
            包含点云数据和域标签的字典
        """
        file_path, domain = self.data_files[idx]
        
        # 加载点云数据
        points = np.load(file_path).astype(np.float32)
        
        # 数据增强
        if self.augment:
            points = self.preprocessor.augment_point_cloud(
                points,
                rotation_range=self.augment_params.get("rotation_range", 0.1),
                jitter_std=self.augment_params.get("jitter_std", 0.01),
                scaling_range=self.augment_params.get("scaling_range", (0.9, 1.1))
            )
        
        # 转换为张量
        points_tensor = torch.from_numpy(points).float()
        
        # 域标签
        domain_label = 0 if domain == "sim" else 1
        
        return {
            "points": points_tensor,
            "domain": torch.tensor(domain_label, dtype=torch.long),
            "domain_name": domain,
            "file_path": file_path
        }
    
    def get_paired_sample(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        获取配对样本（一个sim样本和一个real样本）
        Returns:
            sim样本和real样本的元组
        """
        # 随机选择sim样本
        sim_idx = random.randint(0, len(self.sim_files) - 1)
        sim_file_path, sim_domain = self.sim_files[sim_idx]
        
        # 随机选择real样本
        real_idx = random.randint(0, len(self.real_files) - 1)
        real_file_path, real_domain = self.real_files[real_idx]
        
        # 加载数据
        sim_points = np.load(sim_file_path).astype(np.float32)
        real_points = np.load(real_file_path).astype(np.float32)
        
        # 数据增强
        if self.augment:
            sim_points = self.preprocessor.augment_point_cloud(sim_points, **self.augment_params)
            real_points = self.preprocessor.augment_point_cloud(real_points, **self.augment_params)
        
        sim_sample = {
            "points": torch.from_numpy(sim_points).float(),
            "domain": torch.tensor(0, dtype=torch.long),
            "domain_name": "sim",
            "file_path": sim_file_path
        }
        
        real_sample = {
            "points": torch.from_numpy(real_points).float(),
            "domain": torch.tensor(1, dtype=torch.long),
            "domain_name": "real",
            "file_path": real_file_path
        }
        
        return sim_sample, real_sample


class PairedPointCloudDataset(Dataset):
    """配对点云数据集，确保每个batch包含相同数量的sim和real样本"""
    
    def __init__(self, sim_files: List[str], real_files: List[str],
                 augment: bool = False, augment_params: Optional[Dict] = None):
        """
        初始化配对数据集
        Args:
            sim_files: sim文件路径列表
            real_files: real文件路径列表
            augment: 是否进行数据增强
            augment_params: 数据增强参数
        """
        self.sim_files = sim_files
        self.real_files = real_files
        self.augment = augment
        self.augment_params = augment_params or {}
        
        if self.augment:
            self.preprocessor = PointCloudPreprocessor()
        
        # 确保两个域的样本数量相同
        self.length = min(len(sim_files), len(real_files))
        
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取配对样本
        Args:
            idx: 样本索引
        Returns:
            包含sim和real样本的字典
        """
        # 获取对应的文件
        sim_file = self.sim_files[idx % len(self.sim_files)]
        real_file = self.real_files[idx % len(self.real_files)]
        
        # 加载点云数据
        sim_points = np.load(sim_file).astype(np.float32)
        real_points = np.load(real_file).astype(np.float32)
        
        # 数据增强
        if self.augment:
            sim_points = self.preprocessor.augment_point_cloud(sim_points, **self.augment_params)
            real_points = self.preprocessor.augment_point_cloud(real_points, **self.augment_params)
        
        return {
            "sim_points": torch.from_numpy(sim_points).float(),
            "real_points": torch.from_numpy(real_points).float(),
            "sim_domain": torch.tensor(0, dtype=torch.long),
            "real_domain": torch.tensor(1, dtype=torch.long),
            "sim_file": sim_file,
            "real_file": real_file
        }


def create_data_loaders(data_dir: str, batch_size: int = 8, 
                       num_workers: int = 4, pin_memory: bool = True,
                       augment_train: bool = True, augment_params: Optional[Dict] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        augment_train: 是否对训练数据进行增强
        augment_params: 数据增强参数
    Returns:
        训练、验证、测试数据加载器
    """
    # 加载数据集划分
    splits_file = os.path.join(data_dir, "dataset_splits.pkl")
    with open(splits_file, "rb") as f:
        splits = pickle.load(f)
    
    # 创建数据集
    train_dataset = PointCloudDataset(
        splits["train"], 
        augment=augment_train,
        augment_params=augment_params
    )
    
    val_dataset = PointCloudDataset(
        splits["val"], 
        augment=False
    )
    
    test_dataset = PointCloudDataset(
        splits["test"], 
        augment=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def create_paired_data_loaders(data_dir: str, batch_size: int = 8,
                              num_workers: int = 4, pin_memory: bool = True,
                              augment_train: bool = True, augment_params: Optional[Dict] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建配对数据加载器
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        augment_train: 是否对训练数据进行增强
        augment_params: 数据增强参数
    Returns:
        训练、验证、测试配对数据加载器
    """
    # 加载数据集划分
    splits_file = os.path.join(data_dir, "dataset_splits.pkl")
    with open(splits_file, "rb") as f:
        splits = pickle.load(f)
    
    # 分离不同域的文件
    def separate_domains(file_list):
        sim_files = [path for path, domain in file_list if domain == "sim"]
        real_files = [path for path, domain in file_list if domain == "real"]
        return sim_files, real_files
    
    train_sim, train_real = separate_domains(splits["train"])
    val_sim, val_real = separate_domains(splits["val"])
    test_sim, test_real = separate_domains(splits["test"])
    
    # 创建配对数据集
    train_dataset = PairedPointCloudDataset(
        train_sim, train_real,
        augment=augment_train,
        augment_params=augment_params
    )
    
    val_dataset = PairedPointCloudDataset(
        val_sim, val_real,
        augment=False
    )
    
    test_dataset = PairedPointCloudDataset(
        test_sim, test_real,
        augment=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义batch整理函数
    Args:
        batch: 批次数据列表
    Returns:
        整理后的批次数据
    """
    # 获取批次中所有的键
    keys = batch[0].keys()
    
    # 为每个键创建批次张量
    batch_dict = {}
    for key in keys:
        if key in ["file_path", "sim_file", "real_file", "domain_name"]:
            # 字符串列表不需要堆叠
            batch_dict[key] = [item[key] for item in batch]
        else:
            # 数值张量需要堆叠
            batch_dict[key] = torch.stack([item[key] for item in batch])
    
    return batch_dict