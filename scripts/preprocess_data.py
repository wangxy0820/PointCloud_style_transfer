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

from data.preprocessing import PointCloudPreprocessor

def load_point_cloud(file_path: str) -> np.ndarray:
    """加载点云文件"""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.txt'):
        return np.loadtxt(file_path)
    elif file_path.endswith('.pt'):
        data = torch.load(file_path, weights_only=False)
        # 兼容不同格式的.pt文件
        if isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported .pt content type in {file_path}")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess point cloud data with correct isotropic normalization')
    parser.add_argument('--sim_dir', type=str, required=True, help='Simulation data directory')
    parser.add_argument('--real_dir', type=str, required=True, help='Real world data directory')
    parser.add_argument('--output_dir', type=str, default='datasets/processed', 
                       help='Output directory for processed data')
    parser.add_argument('--chunk_size', type=int, default=4096, help='Chunk size')
    parser.add_argument('--overlap_ratio', type=float, default=0.2, help='Overlap ratio between chunks')
    parser.add_argument('--total_points', type=int, default=120000, help='Total points per point cloud for resampling')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
    
    # --- FIXED: 移除了不再需要的 use_lidar_normalization 参数 ---
    preprocessor = PointCloudPreprocessor(
        total_points=args.total_points,
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap_ratio
    )
    
    print("Preprocessing with correct isotropic normalization:")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Overlap ratio: {args.overlap_ratio}")
    
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
    # 将剩余部分对半分给验证集和测试集
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)
    
    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    
    # 处理每个split
    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_indices)} files)...")
        
        for i, idx in enumerate(tqdm(split_indices, desc=f"Processing {split_name}")):
            try:
                sim_points = load_point_cloud(sim_files[idx])
                real_points = load_point_cloud(real_files[idx])
                
                # 确保点数统一
                if len(sim_points) != args.total_points:
                    if len(sim_points) > args.total_points:
                        indices_to_keep = np.random.choice(len(sim_points), args.total_points, replace=False)
                        sim_points = sim_points[indices_to_keep]
                    else:
                        indices_to_add = np.random.choice(len(sim_points), args.total_points - len(sim_points), replace=True)
                        sim_points = np.concatenate([sim_points, sim_points[indices_to_add]], axis=0)
                
                if len(real_points) != args.total_points:
                    if len(real_points) > args.total_points:
                        indices_to_keep = np.random.choice(len(real_points), args.total_points, replace=False)
                        real_points = real_points[indices_to_keep]
                    else:
                        indices_to_add = np.random.choice(len(real_points), args.total_points - len(real_points), replace=True)
                        real_points = np.concatenate([real_points, real_points[indices_to_add]], axis=0)
                
                # 预处理并保存
                preprocessor.save_preprocessed_data(
                    sim_points=sim_points,
                    real_points=real_points,
                    output_dir=os.path.join(args.output_dir, split_name),
                    file_id=f'{split_name}_{i:04d}'
                )
                
            except Exception as e:
                print(f"\nError processing file pair sim:{sim_files[idx]} real:{real_files[idx]}. Error: {e}")
                continue
    
    print(f"\nPreprocessing completed! Output saved to: {args.output_dir}")
    
    config_path = os.path.join(args.output_dir, 'preprocessing_config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump({
            'chunk_size': args.chunk_size,
            'overlap_ratio': args.overlap_ratio,
            'total_points': args.total_points,
            'normalization_method': 'isotropic',
            'train_files': len(splits['train']),
            'val_files': len(splits['val']),
            'test_files': len(splits['test'])
        }, f, indent=4)
    
    print(f"Configuration saved to: {config_path}")

if __name__ == "__main__":
    main()