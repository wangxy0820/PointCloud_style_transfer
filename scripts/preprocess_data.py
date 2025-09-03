# scripts/preprocess_data.py

import os
import sys
import argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 确保可以正确导入data包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入分层预处理器
from data.preprocessing import PointCloudPreprocessor

def load_point_cloud(file_path: str) -> np.ndarray:
    """加载点云文件"""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.txt'):
        # 增强对不同分隔符的兼容性
        try:
            return np.loadtxt(file_path, delimiter=',')
        except ValueError:
            return np.loadtxt(file_path, delimiter=' ')
    elif file_path.endswith('.pt'):
        data = torch.load(file_path, weights_only=False)
        if isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported .pt content type in {file_path}")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess point cloud data for hierarchical model')
    parser.add_argument('--sim_dir', type=str, required=True, help='Simulation data directory')
    parser.add_argument('--real_dir', type=str, required=True, help='Real world data directory')
    # 输出目录应与config中的 processed_data_dir 保持一致
    parser.add_argument('--output_dir', type=str, default='datasets/processed_hierarchical', 
                       help='Output directory for processed hierarchical data')
    parser.add_argument('--total_points', type=int, default=120000, help='Total points per point cloud for resampling')
    parser.add_argument('--global_points', type=int, default=30000, help='Number of points for global downsampling')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
    
    # 初始化分层预处理器
    preprocessor = PointCloudPreprocessor(
        total_points=args.total_points,
        global_points=args.global_points
    )
    
    print("Preprocessing for hierarchical model:")
    print(f"  Total points: {args.total_points}")
    print(f"  Global points (downsampled): {args.global_points}")
    
    # 获取文件列表
    sim_files = sorted(glob(os.path.join(args.sim_dir, '*')))
    real_files = sorted(glob(os.path.join(args.real_dir, '*')))
    
    if len(sim_files) != len(real_files):
        print(f"Warning: Number of simulation files ({len(sim_files)}) != real files ({len(real_files)})")
        min_files = min(len(sim_files), len(real_files))
        sim_files = sim_files[:min_files]
        real_files = real_files[:min_files]
    
    print(f"Found {len(sim_files)} paired files")
    
    # 分割数据集
    indices = list(range(len(sim_files)))
    train_val_ratio = args.train_ratio
    val_test_ratio = 1.0 - train_val_ratio
    
    train_idx, temp_idx = train_test_split(indices, test_size=val_test_ratio, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)
    
    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    
    # 处理每个split
    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_indices)} files)...")
        
        for i, idx in enumerate(tqdm(split_indices, desc=f"Processing {split_name}")):
            try:
                sim_points = load_point_cloud(sim_files[idx])
                real_points = load_point_cloud(real_files[idx])
                
                # 统一调用分层数据保存方法
                # 这个方法内部会处理重采样、归一化、下采样和保存
                preprocessor.save_hierarchical_data(
                    sim_points=sim_points,
                    real_points=real_points,
                    output_dir=os.path.join(args.output_dir, split_name),
                    file_id=f'{split_name}_{i:04d}'
                )
                
            except Exception as e:
                print(f"\nError processing file pair sim:{sim_files[idx]} real:{real_files[idx]}. Error: {e}")
                continue
    
    print(f"\nPreprocessing completed! Output saved to: {args.output_dir}")
    
    # 保存预处理配置信息
    config_path = os.path.join(args.output_dir, 'preprocessing_config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump({
            'total_points': args.total_points,
            'global_points': args.global_points,
            'normalization_method': 'isotropic',
            'train_files': len(splits['train']),
            'val_files': len(splits['val']),
            'test_files': len(splits['test'])
        }, f, indent=4)
    
    print(f"Configuration saved to: {config_path}")

if __name__ == "__main__":
    main()