#!/usr/bin/env python3
"""
简单的风格转换测试
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders


def simple_style_transfer(source, target, alpha=0.3):
    """最简单的统计风格转换"""
    # 计算统计量
    src_mean = source.mean(dim=1, keepdim=True)
    src_std = source.std(dim=1, keepdim=True)
    
    tgt_mean = target.mean(dim=1, keepdim=True)
    tgt_std = target.std(dim=1, keepdim=True)
    
    # 标准化
    normalized = (source - src_mean) / (src_std + 1e-6)
    
    # 应用目标统计
    transferred = normalized * tgt_std + tgt_mean
    
    # 混合
    return (1 - alpha) * source + alpha * transferred


def test_simple_transfer():
    """测试简单转换"""
    print("Testing Simple Style Transfer")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    test_loader, _, _ = create_dataloaders(
        "datasets/processed",
        batch_size=1,
        num_workers=0,
        chunk_size=4096
    )
    
    batch = next(iter(test_loader))
    sim_points = batch['sim_points'].to(device)
    real_points = batch['real_points'].to(device)
    
    print(f"Sim shape: {sim_points.shape}, range: [{sim_points.min():.3f}, {sim_points.max():.3f}]")
    print(f"Real shape: {real_points.shape}, range: [{real_points.min():.3f}, {real_points.max():.3f}]")
    
    # 统计转换
    with torch.no_grad():
        for alpha in [0.1, 0.3, 0.5, 0.7]:
            transferred = simple_style_transfer(sim_points, real_points, alpha)
            print(f"\nAlpha={alpha}:")
            print(f"  Range: [{transferred.min():.3f}, {transferred.max():.3f}]")
            print(f"  Mean: {transferred.mean():.3f}, Std: {transferred.std():.3f}")
            
            # 保存一个样本
            if alpha == 0.3:
                np.save(f'simple_transfer_alpha_{alpha}.npy', transferred[0].cpu().numpy())
    
    print("\n✓ Simple test completed!")


if __name__ == "__main__":
    test_simple_transfer()