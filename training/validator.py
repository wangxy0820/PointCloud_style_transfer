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
        """
        model.eval()
        style_encoder.eval()
        
        all_metrics = []
        all_diffusion_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                sim_points = batch['sim_points'].to(self.device)
                real_points = batch['real_points'].to(self.device)
                batch_size = sim_points.shape[0]
                
                # 验证输入数据范围
                sim_range = (sim_points.min().item(), sim_points.max().item())
                real_range = (real_points.min().item(), real_points.max().item())
                
                # 第一个batch打印调试信息
                if len(all_metrics) == 0:
                    print(f"\n  Debug - Input ranges: sim={sim_range}, real={real_range}")
                
                # 1. 计算Diffusion损失（与训练一致）
                style_features = style_encoder(real_points).unsqueeze(1)
                
                # 随机采样几个时间步来估算平均损失
                diffusion_losses = []
                for _ in range(5):  # 采样5个时间步
                    t = torch.randint(0, diffusion_process.num_timesteps, (batch_size,), device=self.device)
                    noise = torch.randn_like(sim_points)
                    noisy_points = diffusion_process.q_sample(sim_points, t, noise)
                    predicted_noise = model(noisy_points, t, style_features)
                    
                    # MSE损失（与训练一致）
                    mse_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                    diffusion_losses.append(mse_loss.item())
                
                avg_diffusion_loss = np.mean(diffusion_losses)
                all_diffusion_losses.append(avg_diffusion_loss)
                
                # 2. 生成完整的点云用于评估质量
                if num_inference_steps < diffusion_process.num_timesteps:
                    # DDIM采样（加速推理）
                    generated = self.stable_ddim_sample(
                        model, diffusion_process, 
                        sim_points.shape, style_features,
                        num_inference_steps
                    )
                else:
                    # 完整DDPM采样
                    generated = self.stable_ddpm_sample(
                        model, diffusion_process, 
                        sim_points.shape, style_features
                    )
                
                # 验证生成的范围
                gen_range = (generated.min().item(), generated.max().item())
                if len(all_metrics) == 0:
                    print(f"  Debug - Generated range: {gen_range}")
                
                # 3. 计算评估指标
                batch_metrics = self.compute_metrics_safe(
                    generated, real_points, sim_points
                )
                all_metrics.append(batch_metrics)
        
        # 4. 汇总指标
        aggregated_metrics = self.aggregate_metrics(all_metrics)
        
        # 添加diffusion损失
        aggregated_metrics['val_diffusion_loss'] = np.mean(all_diffusion_losses)
        aggregated_metrics['val_diffusion_loss_std'] = np.std(all_diffusion_losses)
        
        # 使用diffusion损失作为主要的验证损失
        aggregated_metrics['val_loss'] = aggregated_metrics['val_diffusion_loss']
        
        return aggregated_metrics
    
    def stable_ddim_sample(self, model, diffusion_process, shape, style_features,
                          num_inference_steps: int) -> torch.Tensor:
        """
        稳定的DDIM采样
        """
        device = next(model.parameters()).device
        
        # 从标准正态分布开始
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
            
            # 检查预测是否有异常值
            if predicted_noise.abs().max() > 10:
                print(f"  Warning: Large prediction at step {i}: max={predicted_noise.abs().max():.1f}")
                predicted_noise = torch.clamp(predicted_noise, -5, 5)
            
            # DDIM更新步骤
            alpha_t = diffusion_process.alphas_cumprod[t]
            alpha_t_prev = diffusion_process.alphas_cumprod[t_prev]
            
            # 预测x0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / (torch.sqrt(alpha_t) + 1e-8)
            
            # 限制x0的范围在合理区间内
            x0_pred = torch.clamp(x0_pred, -2, 2)
            
            # DDIM更新
            x = torch.sqrt(alpha_t_prev) * x0_pred + \
                torch.sqrt(1 - alpha_t_prev) * predicted_noise
            
            # 限制x的范围，防止数值爆炸
            x = torch.clamp(x, -3, 3)
        
        # 最终限制输出范围
        return torch.clamp(x, -1.5, 1.5)
    
    def stable_ddpm_sample(self, model, diffusion_process, shape, style_features) -> torch.Tensor:
        """
        DDPM采样
        """
        device = next(model.parameters()).device
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for t in reversed(range(diffusion_process.num_timesteps)):
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 使用模型的p_sample方法，但添加范围限制
            x = diffusion_process.p_sample(model, x, batch_t, style_features)
            
            # 每100步限制一次范围
            if t % 100 == 0:
                x = torch.clamp(x, -3, 3)
        
        return torch.clamp(x, -1.5, 1.5)
    
    def compute_metrics_safe(self, generated: torch.Tensor, 
                            real_points: torch.Tensor,
                            sim_points: torch.Tensor) -> Dict[str, float]:
        """安全地计算指标，处理可能的异常值"""
        metrics = {}
        
        # 首先检查并修正任何异常值
        if generated.abs().max() > 10:
            print(f"  Warning: Clamping generated points from max {generated.abs().max():.1f}")
            generated = torch.clamp(generated, -2, 2)
        
        # 生成质量指标 - Chamfer距离
        try:
            cd = self.metrics.chamfer_distance(generated, real_points)
            metrics['chamfer_distance'] = cd.mean().item()
        except:
            metrics['chamfer_distance'] = 100.0  # 失败时的默认值
        
        # 内容保持指标
        try:
            content_cd = self.metrics.chamfer_distance(generated, sim_points)
            metrics['content_preservation'] = content_cd.mean().item()
        except:
            metrics['content_preservation'] = 100.0
        
        # 均匀性指标
        try:
            metrics['uniformity'] = self.metrics.uniformity_score(generated)
        except:
            metrics['uniformity'] = 0.0
        
        # MSE - 确保在合理范围内计算
        # 对于归一化的数据，MSE应该在0-4范围内
        mse_real = torch.nn.functional.mse_loss(generated, real_points).item()
        mse_sim = torch.nn.functional.mse_loss(generated, sim_points).item()
        
        # 如果MSE异常大，说明有问题
        if mse_real > 100:
            print(f"  Warning: MSE too large ({mse_real:.0f}), something is wrong!")
            # 重新计算，限制范围
            gen_clamped = torch.clamp(generated, -2, 2)
            real_clamped = torch.clamp(real_points, -2, 2)
            sim_clamped = torch.clamp(sim_points, -2, 2)
            mse_real = torch.nn.functional.mse_loss(gen_clamped, real_clamped).item()
            mse_sim = torch.nn.functional.mse_loss(gen_clamped, sim_clamped).item()
        
        metrics['mse_to_real'] = mse_real
        metrics['mse_to_sim'] = mse_sim
        
        # 归一化的指标（用于显示）
        metrics['normalized_chamfer'] = min(metrics['chamfer_distance'], 10.0)
        metrics['normalized_content'] = min(metrics['content_preservation'], 10.0)
        
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
        
        return aggregated