#!/usr/bin/env python3
"""
调试验证过程，找出MSE异常大的原因
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders
from models.diffusion_model import DiffusionProcess
from evaluation.metrics import PointCloudMetrics


def debug_validation():
    """调试验证过程"""
    print("="*60)
    print("DEBUGGING VALIDATION PROCESS")
    print("="*60)
    
    # 创建数据加载器
    _, val_loader, _ = create_dataloaders(
        data_dir="datasets/processed",
        batch_size=2,
        num_workers=0,
        chunk_size=2048
    )
    
    # 获取一个批次
    batch = next(iter(val_loader))
    sim_points = batch['sim_points']
    real_points = batch['real_points']
    
    print(f"\nData shapes:")
    print(f"  sim_points: {sim_points.shape}")
    print(f"  real_points: {real_points.shape}")
    
    print(f"\nData ranges:")
    print(f"  sim_points: [{sim_points.min():.3f}, {sim_points.max():.3f}]")
    print(f"  real_points: [{real_points.min():.3f}, {real_points.max():.3f}]")
    
    # 测试不同的MSE计算
    print(f"\nMSE calculations:")
    
    # 1. 直接MSE（这可能是问题所在）
    mse_direct = torch.nn.functional.mse_loss(sim_points, real_points)
    print(f"  Direct MSE: {mse_direct:.6f}")
    
    # 2. 逐元素检查
    diff = (sim_points - real_points) ** 2
    print(f"  Diff squared range: [{diff.min():.6f}, {diff.max():.6f}]")
    print(f"  Diff squared mean: {diff.mean():.6f}")
    
    # 3. 检查是否有NaN或Inf
    print(f"  Contains NaN: {torch.isnan(sim_points).any()} / {torch.isnan(real_points).any()}")
    print(f"  Contains Inf: {torch.isinf(sim_points).any()} / {torch.isinf(real_points).any()}")
    
    # 4. 模拟diffusion生成（全噪声）
    print(f"\nSimulating diffusion generation:")
    noise = torch.randn_like(sim_points)
    print(f"  Noise range: [{noise.min():.3f}, {noise.max():.3f}]")
    
    # 5. 计算噪声和数据的MSE
    mse_noise = torch.nn.functional.mse_loss(noise, real_points)
    print(f"  MSE (noise vs real): {mse_noise:.6f}")
    
    # 6. 检查Chamfer距离
    metrics = PointCloudMetrics(device='cpu')
    cd = metrics.chamfer_distance(sim_points[:1], real_points[:1])
    print(f"\nChamfer distance: {cd.item():.6f}")
    
    # 7. 检查数据加载是否有问题
    print(f"\nChecking data loading:")
    if 'norm_params' in batch:
        norm_params = batch['norm_params']
        print(f"  Norm params in batch: {norm_params}")
    
    # 8. 手动计算期望的MSE范围
    print(f"\nExpected MSE ranges:")
    print(f"  For normalized data [-1,1]: MSE should be 0-4")
    print(f"  For random noise N(0,1): MSE should be ~2")
    
    # 9. 检查是否错误地使用了原始尺度
    if 'norm_params' in batch and 'sim' in batch['norm_params']:
        scale = batch['norm_params']['sim'].get('scale', 1.0)
        if isinstance(scale, torch.Tensor):
            scale = scale[0].item()
        
        print(f"\nChecking scale issues:")
        print(f"  Normalization scale: {scale}")
        print(f"  1/scale: {1/scale}")
        
        # 如果错误地应用了逆变换
        wrong_scale = sim_points / scale
        wrong_mse = torch.nn.functional.mse_loss(wrong_scale, real_points / scale)
        print(f"  MSE with wrong scaling: {wrong_mse:.6f}")
        
        # 这可能解释了巨大的MSE值
        if scale < 0.1:  # scale很小，说明原始数据很大
            print(f"  ⚠️  Scale is very small! Original data might be ~{1/scale:.1f} in magnitude")
            print(f"  ⚠️  If mistakenly using original scale, MSE could be ~{(1/scale)**2:.0f}")


def check_diffusion_process():
    """检查diffusion过程"""
    print("\n" + "="*60)
    print("CHECKING DIFFUSION PROCESS")
    print("="*60)
    
    # 创建diffusion过程
    diffusion = DiffusionProcess(
        num_timesteps=1000,
        beta_schedule='cosine',
        device='cpu'
    )
    
    # 测试数据
    x = torch.randn(2, 100, 3) * 0.5  # 小一点的初始数据
    
    # 测试不同时间步的噪声
    for t in [0, 100, 500, 999]:
        t_tensor = torch.tensor([t, t])
        noise = torch.randn_like(x)
        
        noisy_x = diffusion.q_sample(x, t_tensor, noise)
        
        print(f"\nTimestep {t}:")
        print(f"  Original range: [{x.min():.3f}, {x.max():.3f}]")
        print(f"  Noisy range: [{noisy_x.min():.3f}, {noisy_x.max():.3f}]")
        print(f"  Noise scale: {noisy_x.std():.3f}")


if __name__ == "__main__":
    debug_validation()
    check_diffusion_process()