"""
数据预处理脚本
"""

import argparse
import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch
import shutil

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import ImprovedPointCloudPreprocessor


def process_single_pair(args):
    """处理单对点云文件"""
    sim_file, real_file, output_dir, file_id, preprocessor_params = args
    
    try:
        # 创建预处理器
        preprocessor = ImprovedPointCloudPreprocessor(**preprocessor_params)
        
        # 加载点云
        sim_points = np.load(sim_file).astype(np.float32)
        real_points = np.load(real_file).astype(np.float32)
        
        # 确保形状正确 (N, 3)
        if sim_points.shape[1] != 3:
            if sim_points.shape[0] == 3:
                sim_points = sim_points.T
            else:
                raise ValueError(f"Invalid shape for sim_points: {sim_points.shape}")
                
        if real_points.shape[1] != 3:
            if real_points.shape[0] == 3:
                real_points = real_points.T
            else:
                raise ValueError(f"Invalid shape for real_points: {real_points.shape}")
        
        # 预处理并保存
        save_path = preprocessor.save_preprocessed_data(
            sim_points, real_points, output_dir, file_id
        )
        
        return True, save_path
    
    except Exception as e:
        return False, str(e)


def create_split_folders(output_dir: str, file_paths: list, 
                        train_ratio: float, val_ratio: float):
    """创建数据集划分并复制文件到相应文件夹"""
    
    # 创建划分文件夹
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 随机打乱文件
    np.random.shuffle(file_paths)
    
    # 计算划分点
    n_total = len(file_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分文件
    train_files = file_paths[:n_train]
    val_files = file_paths[n_train:n_train + n_val]
    test_files = file_paths[n_train + n_val:]
    
    # 复制文件到相应目录
    print("\nCreating dataset splits...")
    
    # 复制训练文件
    for i, src_path in enumerate(tqdm(train_files, desc="Copying train files")):
        filename = f"train_{i:04d}.pt"
        dst_path = os.path.join(train_dir, filename)
        shutil.copy2(src_path, dst_path)
    
    # 复制验证文件
    for i, src_path in enumerate(tqdm(val_files, desc="Copying val files")):
        filename = f"val_{i:04d}.pt"
        dst_path = os.path.join(val_dir, filename)
        shutil.copy2(src_path, dst_path)
    
    # 复制测试文件
    for i, src_path in enumerate(tqdm(test_files, desc="Copying test files")):
        filename = f"test_{i:04d}.pt"
        dst_path = os.path.join(test_dir, filename)
        shutil.copy2(src_path, dst_path)
    
    # 保存划分信息
    split_info = {
        'train': train_files,
        'val': val_files,
        'test': test_files,
        'train_count': len(train_files),
        'val_count': len(val_files),
        'test_count': len(test_files),
        'total_count': n_total
    }
    
    split_file = os.path.join(output_dir, 'dataset_split_info.pt')
    torch.save(split_info, split_file)
    
    return len(train_files), len(val_files), len(test_files)


def main():
    parser = argparse.ArgumentParser(description='Preprocess point cloud data')
    parser.add_argument('--sim_dir', required=True, help='Simulation data directory')
    parser.add_argument('--real_dir', required=True, help='Real world data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--chunk_size', type=int, default=2048, help='Points per chunk')
    parser.add_argument('--overlap_ratio', type=float, default=0.3, help='Overlap ratio between chunks')
    parser.add_argument('--total_points', type=int, default=120000, help='Total points in full point cloud')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--temp_dir', type=str, default=None, help='Temporary directory for intermediate files')
    
    args = parser.parse_args()
    
    # 设置临时目录
    if args.temp_dir is None:
        args.temp_dir = os.path.join(args.output_dir, 'temp')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # 获取文件列表
    sim_files = sorted(glob.glob(os.path.join(args.sim_dir, '*.npy')))
    real_files = sorted(glob.glob(os.path.join(args.real_dir, '*.npy')))
    
    print(f"Found {len(sim_files)} simulation files")
    print(f"Found {len(real_files)} real world files")
    
    if len(sim_files) == 0 or len(real_files) == 0:
        print("Error: No .npy files found in the specified directories!")
        return
    
    # 确保数量匹配
    num_pairs = min(len(sim_files), len(real_files))
    sim_files = sim_files[:num_pairs]
    real_files = real_files[:num_pairs]
    
    print(f"Processing {num_pairs} point cloud pairs")
    
    # 准备处理参数
    preprocessor_params = {
        'total_points': args.total_points,
        'chunk_size': args.chunk_size,
        'overlap_ratio': args.overlap_ratio
    }
    
    process_args = []
    for i, (sim_file, real_file) in enumerate(zip(sim_files, real_files)):
        file_id = f'pair_{i:04d}'
        process_args.append((sim_file, real_file, args.temp_dir, file_id, preprocessor_params))
    
    # 并行处理
    print(f"\nProcessing {num_pairs} point cloud pairs with {args.num_workers} workers...")
    
    success_count = 0
    processed_files = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_pair, process_args),
            total=num_pairs,
            desc="Processing"
        ))
    
    # 统计结果
    for i, (success, result) in enumerate(results):
        if success:
            success_count += 1
            processed_files.append(result)
        else:
            print(f"Failed to process pair {i}: {result}")
    
    print(f"\nSuccessfully processed {success_count}/{num_pairs} pairs")
    
    # 创建数据集划分
    if success_count > 0:
        n_train, n_val, n_test = create_split_folders(
            args.output_dir, 
            processed_files,
            args.train_ratio,
            args.val_ratio
        )
        
        print(f"\nDataset split created:")
        print(f"  Train: {n_train} samples ({n_train/success_count*100:.1f}%)")
        print(f"  Val: {n_val} samples ({n_val/success_count*100:.1f}%)")
        print(f"  Test: {n_test} samples ({n_test/success_count*100:.1f}%)")
        
        # 清理临时文件
        print("\nCleaning up temporary files...")
        shutil.rmtree(args.temp_dir)
        
        print(f"\nPreprocessing completed! Data saved to:")
        print(f"  {os.path.join(args.output_dir, 'train')}")
        print(f"  {os.path.join(args.output_dir, 'val')}")
        print(f"  {os.path.join(args.output_dir, 'test')}")
    else:
        print("No files were successfully processed!")


if __name__ == "__main__":
    main()