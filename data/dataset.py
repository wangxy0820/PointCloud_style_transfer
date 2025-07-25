import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from typing import List, Tuple, Dict, Optional
import random
import warnings

from .augmentation import PointCloudAugmentation, create_lidar_augmentation


class PointCloudStyleTransferDataset(Dataset):
    """点云风格转换数据集 - 支持LiDAR模式"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 chunk_size: int = 2048,
                 augment: bool = True,
                 max_chunks_per_sample: int = None,
                 config: Optional[object] = None):  # 添加config参数
        """
        Args:
            data_dir: 预处理数据的根目录
            split: 'train', 'val', 或 'test'
            chunk_size: 每个块的点数
            augment: 是否进行数据增强
            max_chunks_per_sample: 每个样本最多使用多少个块（用于渐进式训练）
            config: 配置对象，用于获取LiDAR特定设置
        """
        self.data_dir = data_dir
        self.split = split
        self.chunk_size = chunk_size
        self.augment = augment
        self.max_chunks_per_sample = max_chunks_per_sample
        self.config = config
        
        # 获取该split的目录
        self.split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory does not exist: {self.split_dir}")
        
        # 加载文件列表
        self.files = self._load_file_list()
        
        if len(self.files) == 0:
            raise ValueError(f"No files found in {self.split_dir}")
        
        # 检查数据的chunk_size是否匹配
        self._check_chunk_size_compatibility()
        
        print(f"Loaded {len(self.files)} files for {split} split")
        print(f"Dataset chunk_size: {self.data_chunk_size}, requested chunk_size: {chunk_size}")
        
        # 数据增强
        if augment and split == 'train':
            if config is not None:
                # 使用LiDAR友好的数据增强
                self.augmentation = create_lidar_augmentation(config)
            else:
                # 使用默认数据增强
                self.augmentation = PointCloudAugmentation(
                    rotation_range=0.05,  # LiDAR友好的参数
                    jitter_std=0.005,
                    scale_range=(0.98, 1.02)
                )
        else:
            self.augmentation = None
    
    def _load_file_list(self) -> List[str]:
        """加载文件列表"""
        pattern = os.path.join(self.split_dir, f'{self.split}_*.pt')
        files = sorted(glob.glob(pattern))
        
        if len(files) == 0:
            # 尝试另一种模式
            pattern = os.path.join(self.split_dir, '*.pt')
            files = sorted(glob.glob(pattern))
        
        return files
    
    def _check_chunk_size_compatibility(self):
        """检查数据的chunk_size是否与请求的匹配"""
        if len(self.files) == 0:
            return
        
        # 加载第一个文件检查
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            sample_data = torch.load(self.files[0], weights_only=False)
        
        # 获取数据的chunk_size
        self.data_chunk_size = sample_data.get('chunk_size', 2048)
        
        # 检查是否使用了LiDAR标准化
        self.use_lidar_normalization = sample_data.get('use_lidar_normalization', False)
        if self.use_lidar_normalization:
            print(f"Dataset was preprocessed with LiDAR normalization")
        
        # 检查第一个chunk的实际大小
        if 'sim_chunks' in sample_data and len(sample_data['sim_chunks']) > 0:
            actual_size = len(sample_data['sim_chunks'][0][0])
            if actual_size != self.data_chunk_size:
                self.data_chunk_size = actual_size
        
        # 如果不匹配，发出警告
        if self.data_chunk_size != self.chunk_size:
            print(f"WARNING: Data was preprocessed with chunk_size={self.data_chunk_size}, "
                  f"but dataset is configured with chunk_size={self.chunk_size}")
            print(f"The dataset will use the preprocessed chunk_size={self.data_chunk_size}")
            # 使用数据中的chunk_size
            self.chunk_size = self.data_chunk_size
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 加载预处理的数据（禁用警告）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                data = torch.load(self.files[idx], weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load {self.files[idx]}: {e}")
        
        # 获取块数据
        sim_chunks = data['sim_chunks']
        real_chunks = data['real_chunks']
        
        # 限制块数（用于渐进式训练）
        if self.max_chunks_per_sample is not None:
            num_chunks = min(len(sim_chunks), len(real_chunks), self.max_chunks_per_sample)
        else:
            num_chunks = min(len(sim_chunks), len(real_chunks))
        
        # 随机选择一个块
        chunk_idx = np.random.randint(0, num_chunks)
        
        # 获取块数据和位置
        sim_chunk, sim_pos = sim_chunks[chunk_idx]
        real_chunk, real_pos = real_chunks[chunk_idx]
        
        # 确保是numpy数组
        if isinstance(sim_chunk, torch.Tensor):
            sim_chunk = sim_chunk.numpy()
        if isinstance(real_chunk, torch.Tensor):
            real_chunk = real_chunk.numpy()
        
        # 转换为张量
        sim_points = torch.from_numpy(sim_chunk).float()
        real_points = torch.from_numpy(real_chunk).float()
        
        # 动态调整点数到目标chunk_size（如果需要）
        sim_points = self._adjust_point_count(sim_points, self.chunk_size)
        real_points = self._adjust_point_count(real_points, self.chunk_size)
        
        # 数据增强
        if self.augmentation is not None:
            sim_points = self.augmentation(sim_points)
            real_points = self.augmentation(real_points)
        
        return {
            'sim_points': sim_points,
            'real_points': real_points,
            'sim_position': sim_pos,
            'real_position': real_pos,
            'chunk_idx': chunk_idx,
            'num_chunks': num_chunks,
            'file_idx': idx,
            'data_chunk_size': self.data_chunk_size,
            'use_lidar_normalization': self.use_lidar_normalization,
            'norm_params': data.get('norm_params', {
                'sim': data.get('sim_norm_params', {}),
                'real': data.get('real_norm_params', {})
            })
        }
    
    def _adjust_point_count(self, points: torch.Tensor, target_size: int) -> torch.Tensor:
        """调整点数到目标大小"""
        current_size = len(points)
        
        if current_size == target_size:
            return points
        
        if current_size > target_size:
            # 随机采样
            indices = np.random.choice(current_size, target_size, replace=False)
            return points[indices]
        else:
            # 重复采样
            indices = np.random.choice(current_size, target_size, replace=True)
            return points[indices]


def create_dataloaders(data_dir: str,
                      batch_size: int,
                      num_workers: int = 4,
                      chunk_size: int = 2048,
                      pin_memory: bool = True,
                      max_chunks_per_sample: int = None,
                      config: Optional[object] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器 - 支持LiDAR配置"""
    
    print(f"Creating dataloaders with requested chunk_size={chunk_size}")
    if config and config.use_lidar_normalization:
        print(f"Using LiDAR-aware data loading")
    
    # 创建数据集
    train_dataset = PointCloudStyleTransferDataset(
        data_dir, 
        split='train', 
        chunk_size=chunk_size, 
        augment=True,
        max_chunks_per_sample=max_chunks_per_sample,
        config=config
    )
    
    val_dataset = PointCloudStyleTransferDataset(
        data_dir, 
        split='val', 
        chunk_size=chunk_size, 
        augment=False,
        max_chunks_per_sample=max_chunks_per_sample,
        config=config
    )
    
    test_dataset = PointCloudStyleTransferDataset(
        data_dir, 
        split='test', 
        chunk_size=chunk_size, 
        augment=False,
        max_chunks_per_sample=max_chunks_per_sample,
        config=config
    )
    
    # 获取实际使用的chunk_size（可能与请求的不同）
    actual_chunk_size = train_dataset.chunk_size
    if actual_chunk_size != chunk_size:
        print(f"Note: Using actual chunk_size={actual_chunk_size} from preprocessed data")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


# 快速测试脚本，确保数据加载正常
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "datasets/processed"
    
    if len(sys.argv) > 2:
        chunk_size = int(sys.argv[2])
    else:
        chunk_size = 2048
    
    print(f"Testing data loading from: {data_dir}")
    print(f"Requested chunk_size: {chunk_size}")
    
    # 创建测试配置
    from config import Config
    test_config = Config()
    test_config.use_lidar_normalization = True
    
    try:
        # 测试创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,
            chunk_size=chunk_size,
            config=test_config
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # 测试加载一个批次
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            print(f"\nSample batch:")
            print(f"  sim_points shape: {batch['sim_points'].shape}")
            print(f"  real_points shape: {batch['real_points'].shape}")
            print(f"  chunk_idx: {batch['chunk_idx']}")
            print(f"  num_chunks: {batch['num_chunks']}")
            print(f"  data_chunk_size: {batch['data_chunk_size'][0]}")
            print(f"  use_lidar_normalization: {batch['use_lidar_normalization'][0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()