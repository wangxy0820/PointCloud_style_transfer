#!/usr/bin/env python3
"""
监控训练进度 - 可视化不同epoch的生成结果
"""

import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference import DiffusionInference


def visualize_epoch_progress(checkpoint_dir, sim_file, real_file, output_dir):
    """可视化不同epoch的生成结果"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    sim_points = np.load(sim_file).astype(np.float32)
    real_points = np.load(real_file).astype(np.float32)
    
    if sim_points.shape[1] != 3:
        sim_points = sim_points.T
    if real_points.shape[1] != 3:
        real_points = real_points.T
    
    # 查找所有checkpoint
    checkpoints = []
    
    # Best model
    if os.path.exists(os.path.join(checkpoint_dir, 'best_model.pth')):
        checkpoints.append(('best', os.path.join(checkpoint_dir, 'best_model.pth')))
    
    # Epoch checkpoints
    for ckpt in sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))):
        epoch = int(ckpt.split('_')[-1].split('.')[0])
        checkpoints.append((f'epoch_{epoch}', ckpt))
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    # 创建图形
    n_checkpoints = len(checkpoints)
    fig = plt.figure(figsize=(5 * n_checkpoints, 15))
    
    # 原始数据（第一行）
    for i in range(n_checkpoints):
        ax = fig.add_subplot(3, n_checkpoints, i + 1, projection='3d')
        
        # 采样显示
        sample_idx = np.random.choice(len(sim_points), min(2000, len(sim_points)), replace=False)
        sim_sample = sim_points[sample_idx]
        
        ax.scatter(sim_sample[:, 0], sim_sample[:, 1], sim_sample[:, 2], 
                  c='blue', s=1, alpha=0.5)
        ax.set_title(f'Original\n({checkpoints[i][0]})')
        ax.set_xlim([-15, 15])
        ax.set_ylim([-15, 15])
        ax.set_zlim([-5, 5])
    
    # 生成结果（第二行）
    for i, (name, ckpt_path) in enumerate(checkpoints):
        print(f"Processing {name}...")
        
        try:
            # 加载模型
            inference = DiffusionInference(ckpt_path, device='cuda')
            
            # 生成
            generated = inference.transfer_style(
                sim_points, 
                real_points,
                use_ddim=True,
                ddim_steps=50,
                return_normalized=True  # 先看归一化空间的结果
            )
            
            # 可视化生成结果（归一化空间）
            ax = fig.add_subplot(3, n_checkpoints, n_checkpoints + i + 1, projection='3d')
            
            sample_idx = np.random.choice(len(generated), min(2000, len(generated)), replace=False)
            gen_sample = generated[sample_idx]
            
            ax.scatter(gen_sample[:, 0], gen_sample[:, 1], gen_sample[:, 2], 
                      c='red', s=1, alpha=0.5)
            ax.set_title(f'Generated (normalized)\n{name}')
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            
            # 计算统计信息
            gen_range = (generated.min(), generated.max())
            gen_std = generated.std()
            
            # 可视化逆归一化结果（第三行）
            generated_denorm = inference.transfer_style(
                sim_points, 
                real_points,
                use_ddim=True,
                ddim_steps=50,
                return_normalized=False  # 逆归一化
            )
            
            ax = fig.add_subplot(3, n_checkpoints, 2 * n_checkpoints + i + 1, projection='3d')
            
            sample_idx = np.random.choice(len(generated_denorm), min(2000, len(generated_denorm)), replace=False)
            gen_sample_denorm = generated_denorm[sample_idx]
            
            ax.scatter(gen_sample_denorm[:, 0], gen_sample_denorm[:, 1], gen_sample_denorm[:, 2], 
                      c='green', s=1, alpha=0.5)
            ax.set_title(f'Generated (denormalized)\nRange: [{gen_range[0]:.2f}, {gen_range[1]:.2f}]')
            ax.set_xlim([-15, 15])
            ax.set_ylim([-15, 15])
            ax.set_zlim([-5, 5])
            
            # 清理内存
            del inference
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_progress.png')
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")
    
    # 创建一个简单的对比图
    if len(checkpoints) >= 2:
        fig2, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
        
        # 原始
        sample_idx = np.random.choice(len(sim_points), 2000, replace=False)
        axes[0].scatter(sim_points[sample_idx, 0], sim_points[sample_idx, 1], 
                       sim_points[sample_idx, 2], c='blue', s=1, alpha=0.5)
        axes[0].set_title('Original Simulation')
        
        # 最新生成（归一化空间）
        try:
            inference = DiffusionInference(checkpoints[-1][1], device='cuda')
            generated = inference.transfer_style(sim_points, real_points, use_ddim=True, ddim_steps=50)
            
            sample_idx = np.random.choice(len(generated), 2000, replace=False)
            axes[1].scatter(generated[sample_idx, 0], generated[sample_idx, 1], 
                           generated[sample_idx, 2], c='red', s=1, alpha=0.5)
            axes[1].set_title(f'Generated ({checkpoints[-1][0]})')
            
            # 真实参考
            sample_idx = np.random.choice(len(real_points), 2000, replace=False)
            axes[2].scatter(real_points[sample_idx, 0], real_points[sample_idx, 1], 
                           real_points[sample_idx, 2], c='green', s=1, alpha=0.5)
            axes[2].set_title('Real Reference')
            
        except Exception as e:
            print(f"Error in comparison: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latest_comparison.png'), dpi=150)


if __name__ == "__main__":
    checkpoint_dir = "experiments/my_experiment/checkpoints"
    sim_file = "datasets/simulation/000000.npy"
    real_file = "datasets/real_world/000000.npy"
    output_dir = "results/progress_monitor"
    
    visualize_epoch_progress(checkpoint_dir, sim_file, real_file, output_dir)