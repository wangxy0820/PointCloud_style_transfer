#!/usr/bin/env python3
"""
基础风格转换测试
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unsupervised_diffusion_model import UnsupervisedPointCloudDiffusionModel, UnsupervisedDiffusionProcess
from data.dataset import create_dataloaders


def statistical_style_transfer(source, target, alpha=0.3):
    """简单的统计风格转换作为基准"""
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


def test_style_transfer():
    """测试风格转换"""
    print("Testing Style Transfer")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载checkpoint
    checkpoint_path = "experiments/test1/checkpoints/latest.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model = UnsupervisedPointCloudDiffusionModel(
            hidden_dims=[128, 256, 512, 1024],
            style_dim=256,
            content_dims=[64, 128, 256]
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        diffusion = UnsupervisedDiffusionProcess(
            num_timesteps=checkpoint['config'].num_timesteps,
            beta_schedule=checkpoint['config'].beta_schedule,
            device=device
        )
    else:
        print("No checkpoint found, using untrained model")
        return
    
    # 加载测试数据
    test_loader, _, _ = create_dataloaders(
        "datasets/processed",
        batch_size=1,
        num_workers=0,
        chunk_size=4096
    )
    
    batch = next(iter(test_loader))
    sim_points = batch['sim_points'].to(device)
    real_points = batch['real_points'].to(device)
    
    print(f"\nInput shapes:")
    print(f"  Sim: {sim_points.shape}, range: [{sim_points.min():.3f}, {sim_points.max():.3f}]")
    print(f"  Real: {real_points.shape}, range: [{real_points.min():.3f}, {real_points.max():.3f}]")
    
    # 1. 统计风格转换（基准）
    with torch.no_grad():
        statistical_result = statistical_style_transfer(sim_points, real_points, alpha=0.3)
        print(f"\nStatistical transfer range: [{statistical_result.min():.3f}, {statistical_result.max():.3f}]")
    
    # 2. Diffusion风格转换（快速版）
    with torch.no_grad():
        print("\nDiffusion style transfer (10 steps)...")
        
        # 提取风格
        real_style = model.style_encoder(real_points)
        sim_content = model.content_encoder(sim_points)
        
        # 快速采样
        generated = diffusion.sample(
            model,
            sim_points.shape,
            style_condition=real_style,
            content_condition=sim_content,
            num_inference_steps=10  # 只用10步
        )
        
        print(f"  Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
        
        # 计算与原始的距离
        mse_to_sim = torch.nn.functional.mse_loss(generated, sim_points).item()
        mse_to_real = torch.nn.functional.mse_loss(generated, real_points).item()
        print(f"  MSE to sim: {mse_to_sim:.3f}, MSE to real: {mse_to_real:.3f}")
    
    # 保存结果比较
    np.save('test_original_sim.npy', sim_points[0].cpu().numpy())
    np.save('test_original_real.npy', real_points[0].cpu().numpy())
    np.save('test_statistical_transfer.npy', statistical_result[0].cpu().numpy())
    np.save('test_diffusion_transfer.npy', generated[0].cpu().numpy())
    
    print("\nSaved results for comparison")
    print("✓ Test completed!")


if __name__ == "__main__":
    test_style_transfer()