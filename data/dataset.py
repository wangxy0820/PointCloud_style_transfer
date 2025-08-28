import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from typing import List, Tuple, Dict, Optional
import warnings

from .augmentation import PointCloudAugmentation, create_lidar_augmentation


class PointCloudStyleTransferDataset(Dataset):
    """点云风格转换数据集 - 简化并强化了结构保持"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 chunk_size: int = 4096,
                 augment: bool = True,
                 max_chunks_per_sample: int = None,
                 config: Optional[object] = None):
        self.data_dir = data_dir
        self.split = split
        self.chunk_size = chunk_size
        self.augment = augment
        self.max_chunks_per_sample = max_chunks_per_sample
        
        self.split_dir = os.path.join(data_dir, split)
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory does not exist: {self.split_dir}")
        
        self.files = self._load_file_list()
        if not self.files:
            raise ValueError(f"No files found in {self.split_dir}")
        
        self._check_data_properties()
        
        print(f"Loaded {len(self.files)} files for {split} split")
        
        if augment and split == 'train' and config:
            self.augmentation = create_lidar_augmentation(config)
        else:
            self.augmentation = None
    
    def _load_file_list(self) -> List[str]:
        pattern = os.path.join(self.split_dir, '*.pt')
        return sorted(glob.glob(pattern))
    
    def _check_data_properties(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            sample_data = torch.load(self.files[0], weights_only=False)
        
        self.data_chunk_size = sample_data.get('chunk_size', self.chunk_size)
        if self.data_chunk_size != self.chunk_size:
            print(f"WARNING: Data chunk_size ({self.data_chunk_size}) differs from requested ({self.chunk_size}). "
                  f"Using data chunk_size.")
            self.chunk_size = self.data_chunk_size
            
        if sample_data.get('use_lidar_normalization', False):
            print("Dataset was preprocessed with LiDAR normalization")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            data = torch.load(self.files[idx], weights_only=False)
        
        sim_chunks = data['sim_chunks']
        real_chunks = data['real_chunks']
        
        num_chunks = min(len(sim_chunks), len(real_chunks))
        if self.max_chunks_per_sample is not None:
            num_chunks = min(num_chunks, self.max_chunks_per_sample)
        
        chunk_idx = np.random.randint(0, num_chunks)
        
        sim_chunk_data, _ = sim_chunks[chunk_idx]
        real_chunk_data, _ = real_chunks[chunk_idx]
        
        # 预处理器应该保证chunk大小正确，不再需要随机调整
        sim_points = torch.from_numpy(sim_chunk_data).float()
        real_points = torch.from_numpy(real_chunk_data).float()
        
        # 断言以确保预处理是正确的
        assert sim_points.shape[0] == self.chunk_size, f"Incorrect sim chunk size in {self.files[idx]}"
        assert real_points.shape[0] == self.chunk_size, f"Incorrect real chunk size in {self.files[idx]}"

        if self.augmentation:
            sim_points = self.augmentation(sim_points)
            real_points = self.augmentation(real_points)
        
        return {
            'sim_points': sim_points,
            'real_points': real_points,
        }

# create_dataloaders 函数保持不变
def create_dataloaders(data_dir: str,
                      batch_size: int,
                      num_workers: int = 4,
                      chunk_size: int = 4096,
                      pin_memory: bool = True,
                      max_chunks_per_sample: int = None,
                      config: Optional[object] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print(f"Creating dataloaders with requested chunk_size={chunk_size}")
    if config and getattr(config, 'use_lidar_normalization', False):
        print(f"Using LiDAR-aware data loading")
    
    train_dataset = PointCloudStyleTransferDataset(
        data_dir, split='train', chunk_size=chunk_size, augment=True,
        max_chunks_per_sample=max_chunks_per_sample, config=config
    )
    val_dataset = PointCloudStyleTransferDataset(
        data_dir, split='val', chunk_size=chunk_size, augment=False,
        max_chunks_per_sample=max_chunks_per_sample, config=config
    )
    test_dataset = PointCloudStyleTransferDataset(
        data_dir, split='test', chunk_size=chunk_size, augment=False,
        max_chunks_per_sample=max_chunks_per_sample, config=config
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False, persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False, persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, test_loader
