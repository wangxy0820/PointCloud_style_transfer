#!/usr/bin/env python3
"""
诊断脚本 - 检查数据预处理问题
"""

import torch
import numpy as np
import glob
import os
import sys
from collections import defaultdict

def diagnose_preprocessed_data(data_dir):
    """诊断预处理数据的问题"""
    print("="*60)
    print("DIAGNOSING PREPROCESSED DATA")
    print("="*60)
    print(f"Data directory: {data_dir}\n")
    
    issues_found = []
    
    # 检查各个split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"✗ {split} directory not found!")
            continue
            
        files = sorted(glob.glob(os.path.join(split_dir, '*.pt')))
        print(f"\n{split.upper()} SET: {len(files)} files")
        
        if not files:
            continue
        
        # 分析chunk sizes
        chunk_sizes = defaultdict(int)
        norm_issues = 0
        point_ranges = []
        
        for i, file in enumerate(files[:10]):  # 检查前10个文件
            try:
                data = torch.load(file, weights_only=False)
                
                # 检查chunk size
                stored_chunk_size = data.get('chunk_size', 'Not stored')
                
                # 检查实际chunk大小
                if 'sim_chunks' in data and data['sim_chunks']:
                    for j, (chunk, _) in enumerate(data['sim_chunks'][:3]):  # 检查前3个chunks
                        actual_size = len(chunk)
                        chunk_sizes[actual_size] += 1
                        
                        # 检查数据范围
                        chunk_array = np.array(chunk) if not isinstance(chunk, np.ndarray) else chunk
                        min_val, max_val = chunk_array.min(), chunk_array.max()
                        point_ranges.append((min_val, max_val))
                        
                        # 检查归一化
                        if abs(min_val) > 10 or abs(max_val) > 10:
                            norm_issues += 1
                
                if i == 0:  # 详细打印第一个文件
                    print(f"\n  Sample file: {os.path.basename(file)}")
                    print(f"  - Stored chunk_size: {stored_chunk_size}")
                    print(f"  - Number of sim chunks: {len(data.get('sim_chunks', []))}")
                    print(f"  - Number of real chunks: {len(data.get('real_chunks', []))}")
                    
                    if 'sim_norm_params' in data:
                        norm_params = data['sim_norm_params']
                        print(f"  - Normalization scale: {norm_params.get('scale', 'N/A')}")
                        print(f"  - Normalization max_dist: {norm_params.get('max_dist', 'N/A')}")
                    
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
                issues_found.append(f"Cannot load {file}")
        
        # 报告chunk size统计
        if chunk_sizes:
            print(f"\n  Chunk size distribution:")
            for size, count in sorted(chunk_sizes.items()):
                print(f"    Size {size}: {count} chunks")
            
            if len(chunk_sizes) > 1:
                issues_found.append(f"{split} set has inconsistent chunk sizes: {list(chunk_sizes.keys())}")
        
        # 报告数据范围
        if point_ranges:
            min_vals = [r[0] for r in point_ranges]
            max_vals = [r[1] for r in point_ranges]
            print(f"\n  Data range:")
            print(f"    Min: {min(min_vals):.3f} to {max(min_vals):.3f}")
            print(f"    Max: {min(max_vals):.3f} to {max(max_vals):.3f}")
            
            if norm_issues > 0:
                issues_found.append(f"{split} set has {norm_issues} unnormalized chunks")
    
    # 总结问题
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if issues_found:
        print("✗ Issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
        
        print("\n✗ RECOMMENDATION: Reprocess the data with fixed preprocessing script")
        print("  Run: bash scripts/reprocess_data.sh")
    else:
        print("✓ No major issues found!")
    
    return len(issues_found) == 0


def check_raw_data(sim_dir, real_dir):
    """检查原始数据"""
    print("\n" + "="*60)
    print("CHECKING RAW DATA")
    print("="*60)
    
    sim_files = sorted(glob.glob(os.path.join(sim_dir, '*.npy')))
    real_files = sorted(glob.glob(os.path.join(real_dir, '*.npy')))
    
    print(f"Simulation files: {len(sim_files)}")
    print(f"Real world files: {len(real_files)}")
    
    if sim_files and real_files:
        # 检查第一个文件
        sim_data = np.load(sim_files[0])
        real_data = np.load(real_files[0])
        
        print(f"\nSample simulation data:")
        print(f"  Shape: {sim_data.shape}")
        print(f"  Range: [{sim_data.min():.3f}, {sim_data.max():.3f}]")
        print(f"  Mean: {sim_data.mean():.3f}, Std: {sim_data.std():.3f}")
        
        print(f"\nSample real world data:")
        print(f"  Shape: {real_data.shape}")
        print(f"  Range: [{real_data.min():.3f}, {real_data.max():.3f}]")
        print(f"  Mean: {real_data.mean():.3f}, Std: {real_data.std():.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "datasets/processed"
    
    # 诊断预处理数据
    success = diagnose_preprocessed_data(data_dir)
    
    # 检查原始数据
    if len(sys.argv) > 2:
        sim_dir = sys.argv[2]
        real_dir = sys.argv[3] if len(sys.argv) > 3 else "datasets/real_world"
        check_raw_data(sim_dir, real_dir)
    
    sys.exit(0 if success else 1)