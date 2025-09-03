# data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from typing import Tuple, Dict, Any
import numpy as np

class HierarchicalPointCloudDataset(Dataset):
    """分层点云数据集 - 支持完整的分层数据加载"""
    
    def __init__(self, processed_dir: str, use_hierarchical: bool = True):
        self.processed_dir = processed_dir
        self.use_hierarchical = use_hierarchical
        self.file_paths = sorted(glob.glob(os.path.join(processed_dir, '*_hierarchical.pt')))
        
        if not self.file_paths:
            raise FileNotFoundError(
                f"No hierarchical data files ('*_hierarchical.pt') found in {processed_dir}. "
                "Please run the preprocess_data.py script first."
            )
            
        print(f"Dataset initialized with {len(self.file_paths)} hierarchical files from {processed_dir}.")
        print(f"Hierarchical mode: {'ON' if use_hierarchical else 'OFF (full points only)'}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        根据use_hierarchical标志返回不同的数据
        """
        file_path = self.file_paths[idx]
        
        try:
            data = torch.load(file_path, weights_only=False)
            
            # 验证数据完整性
            required_keys = ['sim_full', 'real_full']
            if self.use_hierarchical:
                required_keys.extend([
                    'sim_global', 'sim_global_indices', 'sim_norm_params',
                    'real_global', 'real_global_indices', 'real_norm_params'
                ])
            
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise KeyError(f"Missing keys in {file_path}: {missing_keys}")
            
            result = {
                'sim_full': torch.from_numpy(data['sim_full']).float(),
                'real_full': torch.from_numpy(data['real_full']).float(),
            }
            
            if self.use_hierarchical:
                # 添加分层数据
                result.update({
                    'sim_global': torch.from_numpy(data['sim_global']).float(),
                    'real_global': torch.from_numpy(data['real_global']).float(),
                    'sim_global_indices': torch.from_numpy(data['sim_global_indices']).long(),
                    'real_global_indices': torch.from_numpy(data['real_global_indices']).long(),
                    'sim_norm_params': data['sim_norm_params'],  # dict类型，保持原样
                    'real_norm_params': data['real_norm_params'],  # dict类型，保持原样
                    'total_points': data.get('total_points', 120000),
                    'global_points': data.get('global_points', 30000)
                })
            
            return result
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # 返回一个默认的数据项，避免训练中断
            if self.use_hierarchical:
                return self._get_default_hierarchical_item()
            else:
                return self._get_default_simple_item()

    def _get_default_simple_item(self) -> Dict[str, torch.Tensor]:
        """返回默认的简单数据项"""
        return {
            'sim_full': torch.zeros(120000, 3),
            'real_full': torch.zeros(120000, 3),
        }
    
    def _get_default_hierarchical_item(self) -> Dict[str, torch.Tensor]:
        """返回默认的分层数据项"""
        return {
            'sim_full': torch.zeros(120000, 3),
            'real_full': torch.zeros(120000, 3),
            'sim_global': torch.zeros(30000, 3),
            'real_global': torch.zeros(30000, 3),
            'sim_global_indices': torch.arange(30000),
            'real_global_indices': torch.arange(30000),
            'sim_norm_params': {'center': np.zeros(3), 'scale': 1.0, 'method': 'isotropic'},
            'real_norm_params': {'center': np.zeros(3), 'scale': 1.0, 'method': 'isotropic'},
            'total_points': 120000,
            'global_points': 30000
        }


def create_dataloaders(processed_dir: str, batch_size: int, num_workers: int, 
                      use_hierarchical: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    为分层模型创建训练和验证数据加载器
    
    Args:
        processed_dir: 处理后的数据目录
        batch_size: 批大小
        num_workers: 工作线程数
        use_hierarchical: 是否使用分层数据（如果False，只加载完整点云）
    
    Returns:
        训练和验证数据加载器的元组
    """
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Train/Val directories not found in {processed_dir}. "
            "Please run preprocessing first."
        )

    train_dataset = HierarchicalPointCloudDataset(train_dir, use_hierarchical=use_hierarchical)
    val_dataset = HierarchicalPointCloudDataset(val_dir, use_hierarchical=use_hierarchical)

    print(f"Created datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # 自定义collate函数来处理不同类型的数据
    def hierarchical_collate_fn(batch):
        """自定义的batch整理函数"""
        if not batch:
            return {}
            
        # 获取第一个样本来确定数据结构
        first_item = batch[0]
        result = {}
        
        # 处理tensor类型的数据
        tensor_keys = ['sim_full', 'real_full']
        if use_hierarchical:
            tensor_keys.extend(['sim_global', 'real_global', 'sim_global_indices', 'real_global_indices'])
        
        for key in tensor_keys:
            if key in first_item:
                result[key] = torch.stack([item[key] for item in batch])
        
        # 处理非tensor数据（归一化参数等）
        if use_hierarchical:
            for key in ['sim_norm_params', 'real_norm_params', 'total_points', 'global_points']:
                if key in first_item:
                    result[key] = [item[key] for item in batch]
        
        return result

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=hierarchical_collate_fn if use_hierarchical else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=hierarchical_collate_fn if use_hierarchical else None
    )
    
    return train_loader, val_loader