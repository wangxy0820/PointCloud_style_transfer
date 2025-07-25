#!/usr/bin/env python3
"""
深度诊断脚本 - 找出为什么生成的点云完全失去形状
"""

import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unsupervised_diffusion_model import UnsupervisedPointCloudDiffusionModel, UnsupervisedDiffusionProcess
from data.dataset import PointCloudStyleTransferDataset
from torch.utils.data import DataLoader


def deep_diagnosis(checkpoint_path: str, data_dir: str):
    """深度诊断问题"""
    print("Deep Diagnosis of Point Cloud Generation Issues")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据和模型
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = UnsupervisedPointCloudDiffusionModel(
        input_dim=3,
        hidden_dims=[128, 256, 512, 1024],
        time_dim=config.time_embed_dim,
        style_dim=256,
        content_dims=[64, 128, 256]
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion_process = UnsupervisedDiffusionProcess(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule,
        device=device
    )
    
    # 2. 加载一个测试样本
    test_dataset = PointCloudStyleTransferDataset(
        data_dir=data_dir,
        split='test',
        chunk_size=config.chunk_size,
        augment=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    batch = next(iter(test_loader))
    
    sim_points = batch['sim_points'].to(device)
    real_points = batch['real_points'].to(device)
    
    print(f"\n1. Input Data Check:")
    print(f"   Sim shape: {sim_points.shape}")
    print(f"   Sim range: [{sim_points.min():.3f}, {sim_points.max():.3f}]")
    print(f"   Sim mean: {sim_points.mean():.3f}, std: {sim_points.std():.3f}")
    
    # 3. 检查特征提取
    print(f"\n2. Feature Extraction Check:")
    with torch.no_grad():
        style = model.style_encoder(real_points)
        content = model.content_encoder(sim_points)
        
        print(f"   Style shape: {style.shape}, norm: {style.norm():.3f}")
        print(f"   Content shape: {content.shape}, norm: {content.norm():.3f}")
        
        # 检查内容编码器是否保留了空间信息
        content_spatial_var = content.var(dim=2).mean()
        print(f"   Content spatial variance: {content_spatial_var:.3f}")
    
    # 4. 逐步检查生成过程
    print(f"\n3. Step-by-step Generation Check:")
    
    # 4.1 检查不同时间步的去噪
    test_timesteps = [999, 750, 500, 250, 100, 50, 10, 0]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    with torch.no_grad():
        # 从纯噪声开始
        x = torch.randn_like(sim_points)
        
        for idx, t_val in enumerate(test_timesteps):
            if t_val == 999:
                x_t = x.clone()
            else:
                # 单步去噪
                t = torch.tensor([t_val], device=device)
                x_t = diffusion_process.p_sample(model, x, t, style, content)
                x = x_t
            
            # 可视化
            ax = axes[idx]
            pts = x_t[0].cpu().numpy()
            sample_idx = np.random.choice(len(pts), min(1000, len(pts)), replace=False)
            ax.scatter(pts[sample_idx, 0], pts[sample_idx, 1], pts[sample_idx, 2], s=1)
            ax.set_title(f't = {t_val}')
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-3, 3])
            
            print(f"   t={t_val}: range=[{x_t.min():.3f}, {x_t.max():.3f}], std={x_t.std():.3f}")
    
    plt.tight_layout()
    plt.savefig('generation_steps.png')
    print(f"   Saved generation steps to: generation_steps.png")
    
    # 5. 检查模型输出
    print(f"\n4. Model Output Analysis:")
    with torch.no_grad():
        # 测试不同噪声水平
        for noise_level in [0.0, 0.1, 0.5, 1.0]:
            noise = torch.randn_like(sim_points) * noise_level
            noisy_points = sim_points + noise
            
            t = torch.randint(0, 1000, (1,), device=device)
            pred_noise = model(noisy_points, t, style, content)
            
            print(f"   Noise level {noise_level}: pred_noise range=[{pred_noise.min():.3f}, {pred_noise.max():.3f}]")
    
    # 6. 检查内容保持
    print(f"\n5. Content Preservation Test:")
    with torch.no_grad():
        # 不使用风格条件，只用内容
        noise = torch.randn_like(sim_points) * 0.1
        noisy_points = sim_points + noise
        
        # 提取相同的内容
        same_content = model.content_encoder(sim_points)
        
        # 使用相同域的风格（应该保持形状）
        same_style = model.style_encoder(sim_points)
        
        t = torch.tensor([100], device=device)
        pred_noise_same = model(noisy_points, t, same_style, same_content)
        
        denoised = noisy_points - pred_noise_same
        
        mse = torch.nn.functional.mse_loss(denoised, sim_points).item()
        print(f"   Same domain MSE: {mse:.3f}")
    
    # 7. 测试简化版本
    print(f"\n6. Testing Simplified Generation:")
    with torch.no_grad():
        # 直接从输入开始，小步去噪
        x = sim_points + torch.randn_like(sim_points) * 0.3
        
        for step in range(10):
            t = torch.tensor([100 - step * 10], device=device)
            pred_noise = model(x, t, style, content)
            x = x - pred_noise * 0.01  # 小步更新
            
        print(f"   Final range: [{x.min():.3f}, {x.max():.3f}]")
        print(f"   Final std: {x.std():.3f}")
        print(f"   Distance to original: {torch.nn.functional.mse_loss(x, sim_points).item():.3f}")
    
    # 8. 诊断结论
    print(f"\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY:")
    
    if content_spatial_var < 0.01:
        print("✗ Content encoder loses spatial information")
        print("  → Solution: Reduce pooling in content encoder")
    
    if style.norm() > 100:
        print("✗ Style features are too large")
        print("  → Solution: Add normalization to style encoder")
    
    if x.std() > 1.5:
        print("✗ Generation is too dispersed")
        print("  → Solution: Reduce noise schedule or add stronger constraints")
    
    print("\nRECOMMENDED FIXES:")
    print("1. Use a simpler model first - remove style modulation")
    print("2. Start with reconstruction task (same domain)")
    print("3. Use smaller noise levels (beta_start=0.0001, beta_end=0.002)")
    print("4. Add skip connections from input to output")
    print("5. Use position encoding to preserve spatial structure")


def test_simple_diffusion():
    """测试简化的diffusion过程"""
    print("\n" + "=" * 80)
    print("Testing Simplified Diffusion Process")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建简单的测试数据
    # 创建一个简单的形状（球体）
    n_points = 1000
    theta = torch.rand(n_points) * 2 * np.pi
    phi = torch.rand(n_points) * np.pi
    
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    points = torch.stack([x, y, z], dim=1).unsqueeze(0).to(device)
    
    # 测试简单的去噪
    noise_levels = [0.1, 0.3, 0.5, 1.0]
    
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(16, 4), subplot_kw={'projection': '3d'})
    
    for idx, noise_level in enumerate(noise_levels):
        noisy = points + torch.randn_like(points) * noise_level
        
        ax = axes[idx]
        pts = noisy[0].cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
        ax.set_title(f'Noise Level: {noise_level}')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
    
    plt.tight_layout()
    plt.savefig('noise_levels_test.png')
    print("Saved noise level test to: noise_levels_test.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep diagnosis of generation issues')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    
    args = parser.parse_args()
    
    deep_diagnosis(args.checkpoint, args.data_dir)
    test_simple_diffusion()