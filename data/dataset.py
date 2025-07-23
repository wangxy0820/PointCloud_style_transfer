import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from typing import List, Tuple, Dict, Optional
import random

from .augmentation import PointCloudAugmentation


class PointCloudStyleTransferDataset(Dataset):
    """点云风格转换数据集"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 chunk_size: int = 2048,
                 augment: bool = True,
                 max_chunks_per_sample: int = None):
        """
        Args:
            data_dir: 预处理数据的根目录
            split: 'train', 'val', 或 'test'
            chunk_size: 每个块的点数
            augment: 是否进行数据增强
            max_chunks_per_sample: 每个样本最多使用多少个块（用于渐进式训练）
        """
        self.data_dir = data_dir
        self.split = split
        self.chunk_size = chunk_size
        self.augment = augment
        self.max_chunks_per_sample = max_chunks_per_sample
        
        # 获取该split的目录
        self.split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory does not exist: {self.split_dir}")
        
        # 加载文件列表
        self.files = self._load_file_list()
        
        if len(self.files) == 0:
            raise ValueError(f"No files found in {self.split_dir}")
        
        print(f"Loaded {len(self.files)} files for {split} split")
        
        # 数据增强
        if augment and split == 'train':
            self.augmentation = PointCloudAugmentation(
                rotation_range=0.1,
                jitter_std=0.01,
                scale_range=(0.95, 1.05)
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
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 加载预处理的数据
        try:
            data = torch.load(self.files[idx])
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
        
        # 确保点数正确
        if len(sim_points) != self.chunk_size:
            # 调整点数
            if len(sim_points) > self.chunk_size:
                indices = np.random.choice(len(sim_points), self.chunk_size, replace=False)
                sim_points = sim_points[indices]
            else:
                indices = np.random.choice(len(sim_points), self.chunk_size, replace=True)
                sim_points = sim_points[indices]
        
        if len(real_points) != self.chunk_size:
            if len(real_points) > self.chunk_size:
                indices = np.random.choice(len(real_points), self.chunk_size, replace=False)
                real_points = real_points[indices]
            else:
                indices = np.random.choice(len(real_points), self.chunk_size, replace=True)
                real_points = real_points[indices]
        
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
            'norm_params': data.get('norm_params', {
                'sim': data.get('sim_norm_params', {}),
                'real': data.get('real_norm_params', {})
            })
        }


def create_dataloaders(data_dir: str,
                      batch_size: int,
                      num_workers: int = 4,
                      chunk_size: int = 2048,
                      pin_memory: bool = True,
                      max_chunks_per_sample: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 创建数据集
    train_dataset = PointCloudStyleTransferDataset(
        data_dir, 
        split='train', 
        chunk_size=chunk_size, 
        augment=True,
        max_chunks_per_sample=max_chunks_per_sample
    )
    
    val_dataset = PointCloudStyleTransferDataset(
        data_dir, 
        split='val', 
        chunk_size=chunk_size, 
        augment=False,
        max_chunks_per_sample=max_chunks_per_sample
    )
    
    test_dataset = PointCloudStyleTransferDataset(
        data_dir, 
        split='test', 
        chunk_size=chunk_size, 
        augment=False,
        max_chunks_per_sample=max_chunks_per_sample
    )
    
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
    
    print(f"Testing data loading from: {data_dir}")
    
    try:
        # 测试创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,
            chunk_size=2048
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
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()