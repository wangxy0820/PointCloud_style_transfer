"""
验证器模块
"""

import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from evaluation.metrics import PointCloudMetrics


class Validator:
    """模型验证器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.metrics = PointCloudMetrics(device=device)
    
    def validate(self, model, style_encoder, diffusion_process, 
                val_loader, num_inference_steps: int = 50) -> Dict[str, float]:
        """
        验证模型性能
        Args:
            model: Diffusion模型
            style_encoder: 风格编码器
            diffusion_process: Diffusion过程
            val_loader: 验证数据加载器
            num_inference_steps: 推理步数（可以比训练时少）
        Returns:
            验证指标字典
        """
        model.eval()
        style_encoder.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                sim_points = batch['sim_points'].to(self.device)
                real_points = batch['real_points'].to(self.device)
                
                # 提取风格特征
                style_features = style_encoder(real_points).unsqueeze(1)
                
                # 快速推理（使用较少的步数）
                if num_inference_steps < diffusion_process.num_timesteps:
                    # DDIM采样（加速推理）
                    generated = self.ddim_sample(
                        model, diffusion_process, 
                        sim_points.shape, style_features,
                        num_inference_steps
                    )
                else:
                    # 完整DDPM采样
                    generated = diffusion_process.sample(
                        model, sim_points.shape, style_features
                    )
                
                # 计算指标
                batch_metrics = self.compute_metrics(
                    generated, real_points, sim_points
                )
                all_metrics.append(batch_metrics)
        
        # 汇总指标
        aggregated_metrics = self.aggregate_metrics(all_metrics)
        
        return aggregated_metrics
    
    def ddim_sample(self, model, diffusion_process, shape, style_features,
                   num_inference_steps: int) -> torch.Tensor:
        """
        DDIM采样（确定性快速采样）
        """
        device = next(model.parameters()).device
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 选择时间步的子集
        timesteps = torch.linspace(
            diffusion_process.num_timesteps - 1, 0, 
            num_inference_steps, dtype=torch.long, device=device
        )
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            
            # 预测噪声
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x, batch_t, style_features)
            
            # DDIM更新步骤
            alpha_t = diffusion_process.alphas_cumprod[t]
            alpha_t_prev = diffusion_process.alphas_cumprod[t_prev]
            
            # 预测x0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # 计算方差（DDIM中设为0）
            sigma_t = 0  # 确定性采样
            
            # 更新x
            x = torch.sqrt(alpha_t_prev) * x0_pred + \
                torch.sqrt(1 - alpha_t_prev - sigma_t**2) * predicted_noise
        
        return x
    
    def compute_metrics(self, generated: torch.Tensor, 
                       real_points: torch.Tensor,
                       sim_points: torch.Tensor) -> Dict[str, float]:
        """计算单批次的指标"""
        metrics = {}
        
        # 生成质量指标
        cd = self.metrics.chamfer_distance(generated, real_points)
        metrics['chamfer_distance'] = cd.mean().item()
        
        # 内容保持指标
        content_cd = self.metrics.chamfer_distance(generated, sim_points)
        metrics['content_preservation'] = content_cd.mean().item()
        
        # 均匀性指标
        metrics['uniformity'] = self.metrics.uniformity_score(generated)
        
        return metrics
    
    def aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """聚合所有批次的指标"""
        aggregated = {}
        
        # 获取所有指标名称
        metric_names = all_metrics[0].keys()
        
        for name in metric_names:
            values = [m[name] for m in all_metrics]
            aggregated[f'val_{name}'] = np.mean(values)
            aggregated[f'val_{name}_std'] = np.std(values)
        
        # 计算综合损失（用于模型选择）
        aggregated['val_loss'] = (
            aggregated['val_chamfer_distance'] + 
            0.5 * aggregated['val_content_preservation']
        )
        
        return aggregated
