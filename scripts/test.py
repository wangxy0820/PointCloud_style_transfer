#!/usr/bin/env python3


import argparse
import os
import sys
import torch
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
from data.dataset import HierarchicalPointCloudDataset
from evaluation.metrics import PointCloudMetrics
from utils.visualization import PointCloudVisualizer
from torch.utils.data import DataLoader


class Tester:
    """模型测试器"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', output_dir: str = 'test_results'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        self.load_model(checkpoint_path)
        
        # 评估器
        self.metrics = PointCloudMetrics(device=str(self.device))
        self.visualizer = PointCloudVisualizer()
        
    def load_model(self, checkpoint_path: str):
        """加载模型"""
        print(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # 初始化模型
        self.model = PointCloudDiffusionModel(
            input_dim=3,
            hidden_dims=[128, 256, 512, 1024],
            time_dim=self.config.time_embed_dim,
            style_dim=256,
            content_dims=[64, 128, 256]
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 如果有EMA权重，使用EMA权重
        if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            print("Using EMA weights")
            ema_params = checkpoint['ema_state_dict']['shadow_params']
            model_params = list(self.model.parameters())
            for param, ema_param in zip(model_params, ema_params):
                param.data.copy_(ema_param.data)
        
        self.model.eval()
        
        # Diffusion过程
        self.diffusion_process = DiffusionProcess(
            num_timesteps=self.config.num_timesteps,
            beta_schedule=self.config.beta_schedule,
            device=self.device
        )
        
        print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    
    @torch.no_grad()
    def test(self, test_loader, compute_all_metrics: bool = True,
             save_generated: bool = False, save_visualizations: bool = False) -> dict:
        """
        测试模型
        """
        all_metrics = []
        sample_results = []
        
        # 创建保存目录
        if save_generated:
            gen_dir = os.path.join(self.output_dir, 'generated')
            os.makedirs(gen_dir, exist_ok=True)
        
        if save_visualizations:
            vis_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        # 测试循环
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            sim_points = batch['sim_points'].to(self.device)
            real_points = batch['real_points'].to(self.device)
            
            batch_size = sim_points.shape[0]
            
            # 测试两个方向的转换
            batch_metrics = {}
            
            # 1. Sim -> Real
            real_style = self.model.style_encoder(real_points)
            sim_content = self.model.content_encoder(sim_points)
            
            sim_to_real = self.diffusion_process.sample(
                self.model,
                sim_points.shape,
                style_condition=real_style,
                content_condition=sim_content,
                num_inference_steps=50  # 快速推理
            )
            
            # 2. Real -> Sim  
            sim_style = self.model.style_encoder(sim_points)
            real_content = self.model.content_encoder(real_points)
            
            real_to_sim = self.diffusion_process.sample(
                self.model,
                real_points.shape,
                style_condition=sim_style,
                content_condition=real_content,
                num_inference_steps=50
            )
            
            # 计算指标
            if compute_all_metrics:
                # Sim->Real 方向
                cd_s2r = self.metrics.chamfer_distance(sim_to_real, real_points)
                batch_metrics['chamfer_sim_to_real'] = cd_s2r.mean().item()
                
                # Real->Sim 方向
                cd_r2s = self.metrics.chamfer_distance(real_to_sim, sim_points)
                batch_metrics['chamfer_real_to_sim'] = cd_r2s.mean().item()
                
                # 内容保持
                content_s2r = self.metrics.chamfer_distance(sim_to_real, sim_points)
                content_r2s = self.metrics.chamfer_distance(real_to_sim, real_points)
                batch_metrics['content_preservation'] = (content_s2r.mean().item() + content_r2s.mean().item()) / 2
                
                # 循环一致性（可选，计算成本较高）
                if batch_idx < 5:  # 只对前几个batch计算
                    # Sim -> Real -> Sim
                    real_to_sim_style = self.model.style_encoder(sim_to_real)
                    real_to_sim_content = self.model.content_encoder(sim_to_real)
                    cycle_sim = self.diffusion_process.sample(
                        self.model,
                        sim_to_real.shape,
                        style_condition=sim_style,
                        content_condition=real_to_sim_content,
                        num_inference_steps=25  # 更快的推理
                    )
                    cycle_loss = self.metrics.chamfer_distance(cycle_sim, sim_points)
                    batch_metrics['cycle_consistency'] = cycle_loss.mean().item()
            
            all_metrics.append(batch_metrics)
            
            # 保存生成的点云
            if save_generated:
                for i in range(batch_size):
                    idx = batch_idx * test_loader.batch_size + i
                    
                    np.save(
                        os.path.join(gen_dir, f'sim_to_real_{idx:04d}.npy'),
                        sim_to_real[i].cpu().numpy()
                    )
                    np.save(
                        os.path.join(gen_dir, f'real_to_sim_{idx:04d}.npy'),
                        real_to_sim[i].cpu().numpy()
                    )
                    np.save(
                        os.path.join(gen_dir, f'original_sim_{idx:04d}.npy'),
                        sim_points[i].cpu().numpy()
                    )
                    np.save(
                        os.path.join(gen_dir, f'original_real_{idx:04d}.npy'),
                        real_points[i].cpu().numpy()
                    )
            
            # 保存可视化
            if save_visualizations and batch_idx < 10:
                for i in range(min(batch_size, 2)):
                    idx = batch_idx * test_loader.batch_size + i
                    
                    # Sim to Real 可视化
                    self.visualizer.plot_style_transfer_result(
                        sim_points[i].cpu().numpy(),
                        sim_to_real[i].cpu().numpy(),
                        real_points[i].cpu().numpy(),
                        title=f'Test Sample {idx} - Sim to Real',
                        save_path=os.path.join(vis_dir, f'sample_{idx:04d}_s2r.png')
                    )
                    
                    # Real to Sim 可视化
                    self.visualizer.plot_style_transfer_result(
                        real_points[i].cpu().numpy(),
                        real_to_sim[i].cpu().numpy(),
                        sim_points[i].cpu().numpy(),
                        title=f'Test Sample {idx} - Real to Sim',
                        save_path=os.path.join(vis_dir, f'sample_{idx:04d}_r2s.png')
                    )
            
            # 保存一些样本结果
            if batch_idx < 5:
                sample_results.append({
                    'batch_idx': batch_idx,
                    'metrics': batch_metrics
                })
        
        # 计算平均指标
        average_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    average_metrics[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        # 汇总结果
        results = {
            'average_metrics': {k: v['mean'] for k, v in average_metrics.items()},
            'detailed_metrics': average_metrics,
            'all_batch_metrics': all_metrics,
            'sample_results': sample_results,
            'num_samples': len(test_loader.dataset),
            'config': vars(self.config) if hasattr(self.config, '__dict__') else str(self.config)
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Test Point Cloud Style Transfer Model')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data directory')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Directory to save test results')
    parser.add_argument('--save_generated', action='store_true',
                       help='Save generated point clouds')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of samples to test (-1 for all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    # 评估选项
    parser.add_argument('--compute_all_metrics', action='store_true',
                       help='Compute all evaluation metrics')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存测试配置
    with open(os.path.join(output_dir, 'test_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading checkpoint...")
    # 加载检查点获取配置
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint['config']
    
    # 创建测试器
    tester = Tester(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=output_dir
    )
    
    # 创建测试数据集
    print("Loading test dataset...")
    test_dataset = HierarchicalPointCloudDataset(
        processed_dir = os.path.join(args.test_data, 'test'),
        use_hierarchical=True
    )
    
    if args.num_samples > 0:
        # 限制测试样本数
        indices = list(range(min(args.num_samples, len(test_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Testing on {len(test_dataset)} samples...")
    
    # 运行测试
    results = tester.test(
        test_loader,
        compute_all_metrics=args.compute_all_metrics,
        save_generated=args.save_generated,
        save_visualizations=args.save_visualizations
    )
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for metric_name, metric_value in results['average_metrics'].items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.6f}")
        else:
            print(f"{metric_name}: {metric_value}")
    
    print("="*60)
    
    # 保存详细结果
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()