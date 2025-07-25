#!/usr/bin/env python3
"""
修复后的重建测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unsupervised_diffusion_model import UnsupervisedPointCloudDiffusionModel, UnsupervisedDiffusionProcess
from data.dataset import create_dataloaders


def test_simple_reconstruction():
    """测试简单重建"""
    print("Testing Simple Reconstruction")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = UnsupervisedPointCloudDiffusionModel(
        hidden_dims=[128, 256, 512, 1024],
        style_dim=256,
        content_dims=[64, 128, 256]
    ).to(device)
    
    # 创建简单的Diffusion过程
    diffusion = UnsupervisedDiffusionProcess(
        num_timesteps=100,  # 减少步数
        beta_schedule="linear",
        device=device
    )
    
    # 加载一个batch测试
    train_loader, _, _ = create_dataloaders(
        "datasets/processed",
        batch_size=1,
        num_workers=0,
        chunk_size=4096  # 使用实际的chunk size
    )
    
    batch = next(iter(train_loader))
    points = batch['sim_points'].to(device)
    
    print(f"Input shape: {points.shape}")
    print(f"Input range: [{points.min():.3f}, {points.max():.3f}]")
    
    # 测试编码器
    with torch.no_grad():
        style = model.style_encoder(points)
        content = model.content_encoder(points)
        
        print(f"\nEncoder outputs:")
        print(f"  Style shape: {style.shape}, norm: {style.norm():.3f}")
        print(f"  Content shape: {content.shape}, norm: {content.norm():.3f}")
        print(f"  Content spatial variance: {content.var(dim=2).mean():.4f}")
        
        # 如果内容方差太小，说明空间信息丢失
        if content.var(dim=2).mean() < 0.1:
            print("  WARNING: Content encoder loses spatial information!")
    
    # 测试模型前向传播（不进行完整的去噪）
    print("\nTesting model forward pass...")
    try:
        with torch.no_grad():
            # 使用较小的时间步
            t = torch.tensor([10], device=device)
            
            # 直接测试模型输出
            output = model(points, t, style, content)
            
            print(f"  Model output shape: {output.shape}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
            
            # 计算简单的MSE
            mse = nn.functional.mse_loss(output, points).item()
            print(f"  Direct MSE: {mse:.4f}")
            
    except Exception as e:
        print(f"  Error in forward pass: {e}")
        print("  This might be due to dimension mismatch in style modulation")
        return
    
    # 简单的统计测试
    print("\nTesting statistical properties...")
    with torch.no_grad():
        # 计算输入输出的统计差异
        input_mean = points.mean().item()
        input_std = points.std().item()
        output_mean = output.mean().item()
        output_std = output.std().item()
        
        print(f"  Input: mean={input_mean:.3f}, std={input_std:.3f}")
        print(f"  Output: mean={output_mean:.3f}, std={output_std:.3f}")
    
    print("\n✓ Test completed!")


if __name__ == "__main__":
    test_simple_reconstruction()