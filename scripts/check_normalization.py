#!/usr/bin/env python3
"""
检查数据是否正确归一化
"""

import torch
import numpy as np
import glob
import os
import sys


def check_normalization(data_dir):
    """检查预处理数据的归一化情况"""
    print("="*60)
    print("CHECKING DATA NORMALIZATION")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        files = sorted(glob.glob(os.path.join(split_dir, '*.pt')))[:5]  # 检查前5个文件
        
        if not files:
            continue
            
        print(f"\n{split.upper()} SET:")
        
        all_ranges = []
        
        for file in files:
            data = torch.load(file, weights_only=False)
            
            # 检查归一化参数
            if 'sim_norm_params' in data:
                sim_params = data['sim_norm_params']
                print(f"\n  File: {os.path.basename(file)}")
                print(f"    Sim normalization:")
                print(f"      Center: {sim_params.get('center', 'N/A')}")
                print(f"      Scale: {sim_params.get('scale', 'N/A')}")
                print(f"      Max dist: {sim_params.get('max_dist', 'N/A')}")
            
            # 检查实际数据范围
            if 'sim_chunks' in data and data['sim_chunks']:
                # 检查前3个chunks
                for i, (chunk, _) in enumerate(data['sim_chunks'][:3]):
                    chunk_array = np.array(chunk) if not isinstance(chunk, np.ndarray) else chunk
                    min_val = chunk_array.min()
                    max_val = chunk_array.max()
                    max_abs = np.abs(chunk_array).max()
                    
                    all_ranges.append((min_val, max_val))
                    
                    if i == 0:  # 只打印第一个chunk
                        print(f"    First chunk range: [{min_val:.3f}, {max_val:.3f}], max_abs: {max_abs:.3f}")
                        
                        # 检查是否在[-1, 1]范围内
                        if max_abs > 1.0 + 1e-6:
                            print(f"    ⚠️  WARNING: Data not properly normalized! Expected [-1, 1]")
                        else:
                            print(f"    ✓ Data is properly normalized")
        
        # 汇总统计
        if all_ranges:
            min_vals = [r[0] for r in all_ranges]
            max_vals = [r[1] for r in all_ranges]
            print(f"\n  Overall range for {split}:")
            print(f"    Min: {min(min_vals):.3f} to {max(min_vals):.3f}")
            print(f"    Max: {min(max_vals):.3f} to {max(max_vals):.3f}")
            
            overall_max_abs = max(abs(min(min_vals)), abs(max(max_vals)))
            if overall_max_abs > 1.0 + 1e-6:
                print(f"    ⚠️  Data is NOT normalized to [-1, 1]")
                return False
            else:
                print(f"    ✓ Data is normalized to [-1, 1]")
    
    return True


def check_raw_vs_normalized(data_dir):
    """比较原始数据和归一化后的数据"""
    print("\n" + "="*60)
    print("COMPARING RAW VS NORMALIZED")
    print("="*60)
    
    # 加载一个训练文件
    train_files = glob.glob(os.path.join(data_dir, 'train', '*.pt'))
    if not train_files:
        print("No training files found")
        return
    
    data = torch.load(train_files[0], weights_only=False)
    
    if 'sim_norm_params' in data:
        params = data['sim_norm_params']
        print(f"\nNormalization parameters:")
        print(f"  Original max distance: {params.get('max_dist', 'N/A'):.3f}")
        print(f"  Scale factor: {params.get('scale', 'N/A'):.6f}")
        print(f"  Expected range after norm: [-1, 1]")
        
        # 验证计算
        if 'max_dist' in params and 'scale' in params:
            expected_scale = 1.0 / params['max_dist']
            print(f"  Computed scale check: {expected_scale:.6f}")
            print(f"  Scale correct: {abs(expected_scale - params['scale']) < 1e-6}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "datasets/processed"
    
    print(f"Checking data in: {data_dir}\n")
    
    # 检查归一化
    is_normalized = check_normalization(data_dir)
    
    # 比较原始和归一化数据
    check_raw_vs_normalized(data_dir)
    
    # 总结
    print("\n" + "="*60)
    if is_normalized:
        print("✅ RESULT: Data appears to be properly normalized")
    else:
        print("❌ RESULT: Data is NOT properly normalized!")
        print("\nRECOMMENDATION:")
        print("1. Reprocess the data with the fixed preprocessing script")
        print("2. Make sure the normalization scales points to [-1, 1] range")