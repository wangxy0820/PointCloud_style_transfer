"""
模型测试器
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Optional, Tuple  # 添加完整的typing导入
from tqdm import tqdm
import json

from models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
from models.pointnet2_encoder import PointNet2Encoder
from evaluation.metrics import PointCloudMetrics
from utils.visualization import PointCloudVisualizer


class DiffusionTester:
    """Diffusion模型测试器"""
    
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
            time_dim=self.config.time_embed_dim
        ).to(self.device)
        
        self.style_encoder = PointNet2Encoder(
            input_channels=3,
            feature_dim=1024
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.style_encoder.load_state_dict(checkpoint['style_encoder_state_dict'])
        
        # 如果有EMA权重，使用EMA权重
        if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            print("Using EMA weights")
            ema_params = checkpoint['ema_state_dict']['shadow_params']
            model_params = list(self.model.parameters())
            for param, ema_param in zip(model_params, ema_params):
                param.data.copy_(ema_param.data)
        
        self.model.eval()
        self.style_encoder.eval()
        
        # Diffusion过程
        self.diffusion_process = DiffusionProcess(
            num_timesteps=self.config.num_timesteps,
            beta_schedule=self.config.beta_schedule,
            device=self.device
        )
        
    
    @torch.no_grad()
    def test(self, test_loader, compute_all_metrics: bool = True,
             save_generated: bool = False, save_visualizations: bool = False,
             metrics_to_compute: List[str] = None) -> Dict:
        """
        测试模型
        Args:
            test_loader: 测试数据加载器
            compute_all_metrics: 是否计算所有指标
            save_generated: 是否保存生成的点云
            save_visualizations: 是否保存可视化
            metrics_to_compute: 要计算的指标列表
        Returns:
            测试结果字典
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
            
            # 提取风格特征
            style_features = self.style_encoder(real_points).unsqueeze(1)
            
            # 生成
            shape = sim_points.shape
            generated = self.diffusion_process.sample(
                self.model,
                shape,
                style_features
            )
            
            # 计算指标
            batch_metrics = {}
            
            if compute_all_metrics or (metrics_to_compute and 'chamfer' in metrics_to_compute):
                cd = self.metrics.chamfer_distance(generated, real_points)
                batch_metrics['chamfer_distance'] = cd.mean().item()
                batch_metrics['chamfer_distance_std'] = cd.std().item()
            
            if compute_all_metrics or (metrics_to_compute and 'emd' in metrics_to_compute):
                emd = self.metrics.earth_mover_distance(generated, real_points)
                batch_metrics['emd'] = emd.mean().item()
                batch_metrics['emd_std'] = emd.std().item()
            
            if compute_all_metrics or (metrics_to_compute and 'coverage' in metrics_to_compute):
                coverage = self.metrics.coverage_score(generated, real_points)
                batch_metrics['coverage'] = coverage
            
            if compute_all_metrics or (metrics_to_compute and 'uniformity' in metrics_to_compute):
                uniformity = self.metrics.uniformity_score(generated)
                batch_metrics['uniformity'] = uniformity
            
            # 内容保持指标
            if compute_all_metrics or (metrics_to_compute and 'content' in metrics_to_compute):
                content_cd = self.metrics.chamfer_distance(generated, sim_points)
                batch_metrics['content_preservation'] = content_cd.mean().item()
            
            all_metrics.append(batch_metrics)
            
            # 保存生成的点云
            if save_generated:
                for i in range(batch_size):
                    idx = batch_idx * test_loader.batch_size + i
                    
                    np.save(
                        os.path.join(gen_dir, f'generated_{idx:04d}.npy'),
                        generated[i].cpu().numpy()
                    )
                    np.save(
                        os.path.join(gen_dir, f'original_{idx:04d}.npy'),
                        sim_points[i].cpu().numpy()
                    )
                    np.save(
                        os.path.join(gen_dir, f'reference_{idx:04d}.npy'),
                        real_points[i].cpu().numpy()
                    )
            
            # 保存可视化
            if save_visualizations and batch_idx < 10:  # 只保存前10个批次
                for i in range(min(batch_size, 2)):  # 每批最多2个
                    idx = batch_idx * test_loader.batch_size + i
                    
                    self.visualizer.plot_style_transfer_result(
                        sim_points[i].cpu().numpy(),
                        generated[i].cpu().numpy(),
                        real_points[i].cpu().numpy(),
                        title=f'Test Sample {idx}',
                        save_path=os.path.join(vis_dir, f'sample_{idx:04d}.png')
                    )
            
            # 保存一些样本结果用于分析
            if batch_idx < 5:
                sample_results.append({
                    'batch_idx': batch_idx,
                    'metrics': batch_metrics
                })
        
        # 计算平均指标
        average_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
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
    
    def test_single_pointcloud(self, sim_points: np.ndarray, 
                             real_reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        测试单个点云
        Args:
            sim_points: 仿真点云 [N, 3]
            real_reference: 真实参考点云 [M, 3]
        Returns:
            生成的点云和评估指标
        """
        # 转换为张量
        sim_tensor = torch.from_numpy(sim_points).float().unsqueeze(0).to(self.device)
        real_tensor = torch.from_numpy(real_reference).float().unsqueeze(0).to(self.device)
        
        # 提取风格特征
        with torch.no_grad():
            style_features = self.style_encoder(real_tensor).unsqueeze(1)
            
            # 生成
            generated = self.diffusion_process.sample(
                self.model,
                sim_tensor.shape,
                style_features
            )
        
        # 计算指标
        metrics = {
            'chamfer_distance': self.metrics.chamfer_distance(generated, real_tensor).item(),
            'content_preservation': self.metrics.chamfer_distance(generated, sim_tensor).item()
        }
        
        return generated[0].cpu().numpy(), metrics
