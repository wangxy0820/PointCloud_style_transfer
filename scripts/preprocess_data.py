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


def verify_chunk_consistency(data_path):
    """验证chunk大小的一致性"""
    data = torch.load(data_path, weights_only=False)
    
    chunk_sizes = set()
    if 'sim_chunks' in data:
        for chunk, _ in data['sim_chunks']:
            chunk_sizes.add(len(chunk))
    if 'real_chunks' in data:
        for chunk, _ in data['real_chunks']:
            chunk_sizes.add(len(chunk))
    
    expected_size = data.get('chunk_size', 2048)
    
    return len(chunk_sizes) == 1 and expected_size in chunk_sizes, chunk_sizes


def process_single_pair(args):
    """处理单对点云文件"""
    sim_file, real_file, output_dir, file_id, preprocessor_params = args
    
    try:
        # 创建预处理器（使用修复版）
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
        
        print(f"Processing {file_id}: sim={sim_points.shape}, real={real_points.shape}")
        
        # 预处理并保存
        save_path = preprocessor.save_preprocessed_data(
            sim_points, real_points, output_dir, file_id
        )
        
        # 验证处理结果
        is_consistent, sizes = verify_chunk_consistency(save_path)
        if not is_consistent:
            raise ValueError(f"Chunk sizes not consistent: {sizes}")
        
        return True, save_path
    
    except Exception as e:
        return False, str(e)


def create_split_folders(output_dir: str, file_paths: list, 
                        train_ratio: float, val_ratio: float):
    """创建数据集划分"""
    
    # 创建划分文件夹
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 随机打乱文件
    np.random.seed(42)  # 固定种子以保证可重复性
    np.random.shuffle(file_paths)
    
    # 计算划分点
    n_total = len(file_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分文件
    train_files = file_paths[:n_train]
    val_files = file_paths[n_train:n_train + n_val]
    test_files = file_paths[n_train + n_val:]
    
    print("\nCreating dataset splits...")
    
    # 复制并验证训练文件
    train_issues = 0
    for i, src_path in enumerate(tqdm(train_files, desc="Processing train files")):
        filename = f"train_{i:04d}.pt"
        dst_path = os.path.join(train_dir, filename)
        shutil.copy2(src_path, dst_path)
        
        # 验证
        is_consistent, sizes = verify_chunk_consistency(dst_path)
        if not is_consistent:
            train_issues += 1
            print(f"\nWarning: {filename} has inconsistent chunks: {sizes}")
    
    # 复制验证文件
    val_issues = 0
    for i, src_path in enumerate(tqdm(val_files, desc="Processing val files")):
        filename = f"val_{i:04d}.pt"
        dst_path = os.path.join(val_dir, filename)
        shutil.copy2(src_path, dst_path)
        
        is_consistent, sizes = verify_chunk_consistency(dst_path)
        if not is_consistent:
            val_issues += 1
    
    # 复制测试文件
    test_issues = 0
    for i, src_path in enumerate(tqdm(test_files, desc="Processing test files")):
        filename = f"test_{i:04d}.pt"
        dst_path = os.path.join(test_dir, filename)
        shutil.copy2(src_path, dst_path)
        
        is_consistent, sizes = verify_chunk_consistency(dst_path)
        if not is_consistent:
            test_issues += 1
    
    # 保存划分信息
    split_info = {
        'train': train_files,
        'val': val_files,
        'test': test_files,
        'train_count': len(train_files),
        'val_count': len(val_files),
        'test_count': len(test_files),
        'total_count': n_total,
        'train_issues': train_issues,
        'val_issues': val_issues,
        'test_issues': test_issues
    }
    
    split_file = os.path.join(output_dir, 'dataset_split_info.pt')
    torch.save(split_info, split_file)
    
    return len(train_files), len(val_files), len(test_files), train_issues + val_issues + test_issues


def main():
    parser = argparse.ArgumentParser(description='Reprocess point cloud data with fixed chunk sizes')
    parser.add_argument('--sim_dir', required=True, help='Simulation data directory')
    parser.add_argument('--real_dir', required=True, help='Real world data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--chunk_size', type=int, default=2048, help='Points per chunk')
    parser.add_argument('--overlap_ratio', type=float, default=0.3, help='Overlap ratio')
    parser.add_argument('--total_points', type=int, default=120000, help='Total points')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--backup_old', action='store_true', help='Backup old processed data')
    
    args = parser.parse_args()
    
    # 备份旧数据
    if args.backup_old and os.path.exists(args.output_dir):
        backup_dir = f"{args.output_dir}_backup_{np.random.randint(10000)}"
        print(f"Backing up old data to {backup_dir}")
        shutil.move(args.output_dir, backup_dir)
    
    # 设置临时目录
    temp_dir = os.path.join(args.output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 获取文件列表
    sim_files = sorted(glob.glob(os.path.join(args.sim_dir, '*.npy')))
    real_files = sorted(glob.glob(os.path.join(args.real_dir, '*.npy')))
    
    print(f"Found {len(sim_files)} simulation files")
    print(f"Found {len(real_files)} real world files")
    
    if len(sim_files) == 0 or len(real_files) == 0:
        print("Error: No .npy files found!")
        return
    
    # 确保数量匹配
    num_pairs = min(len(sim_files), len(real_files))
    sim_files = sim_files[:num_pairs]
    real_files = real_files[:num_pairs]
    
    print(f"\nProcessing {num_pairs} point cloud pairs")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap ratio: {args.overlap_ratio}")
    
    # 准备处理参数
    preprocessor_params = {
        'total_points': args.total_points,
        'chunk_size': args.chunk_size,
        'overlap_ratio': args.overlap_ratio
    }
    
    process_args = []
    for i, (sim_file, real_file) in enumerate(zip(sim_files, real_files)):
        file_id = f'pair_{i:04d}'
        process_args.append((sim_file, real_file, temp_dir, file_id, preprocessor_params))
    
    # 并行处理
    print(f"\nProcessing with {args.num_workers} workers...")
    
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
        n_train, n_val, n_test, n_issues = create_split_folders(
            args.output_dir, 
            processed_files,
            args.train_ratio,
            args.val_ratio
        )
        
        print(f"\nDataset split created:")
        print(f"  Train: {n_train} samples")
        print(f"  Val: {n_val} samples")
        print(f"  Test: {n_test} samples")
        
        if n_issues > 0:
            print(f"  ⚠️  Warning: {n_issues} files have chunk size issues!")
        else:
            print(f"  ✓ All files have consistent chunk sizes!")
        
        # 清理临时文件
        print("\nCleaning up...")
        shutil.rmtree(temp_dir)
        
        print(f"\n✓ Preprocessing completed!")
        print(f"Data saved to: {args.output_dir}")
        
        # 最终验证
        print("\nFinal verification:")
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(args.output_dir, split)
            files = glob.glob(os.path.join(split_dir, '*.pt'))
            if files:
                data = torch.load(files[0], weights_only=False)
                print(f"  {split}: chunk_size={data.get('chunk_size')}, "
                      f"first chunk size={len(data['sim_chunks'][0][0])}")
    else:
        print("No files were successfully processed!")


if __name__ == "__main__":
    main()
