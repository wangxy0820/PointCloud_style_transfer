"""
无监督Diffusion模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from typing import Dict, Optional, List, Tuple
import numpy as np

from models.unsupervised_diffusion_model import UnsupervisedPointCloudDiffusionModel, UnsupervisedDiffusionProcess
from models.losses import GeometryPreservingDiffusionLoss
from evaluation.metrics import PointCloudMetrics
from utils.visualization import PointCloudVisualizer
from utils.logger import Logger
from utils.checkpoint import CheckpointManager
from training.validator import Validator


class UnsupervisedDiffusionTrainer:
    """无监督Diffusion模型训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # 初始化日志
        self.logger = Logger(
            name='UnsupervisedDiffusionTrainer',
            log_dir=config.log_dir
        )
        
        # 初始化模型
        self.logger.info("Initializing unsupervised models...")
        self.model = UnsupervisedPointCloudDiffusionModel(
            input_dim=3,
            hidden_dims=[128, 256, 512, 1024],
            time_dim=config.time_embed_dim,
            style_dim=256,
            content_dims=[64, 128, 256]
        ).to(self.device)
        
        # Diffusion过程
        self.diffusion_process = UnsupervisedDiffusionProcess(
            num_timesteps=config.num_timesteps,
            beta_schedule=config.beta_schedule,
            device=self.device
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # 损失函数
        self.criterion = GeometryPreservingDiffusionLoss(
            lambda_diffusion=1.0,
            lambda_shape=2.0,      # 形状保持最重要
            lambda_local=1.0,      # 局部结构也重要
            lambda_style=0.1,      # 风格次要
            lambda_smooth=0.5
        )
        
        # EMA
        if config.ema_decay > 0:
            from training.trainer import ExponentialMovingAverage
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=config.ema_decay
            )
        else:
            self.ema = None
        
        # 评估器
        self.metrics = PointCloudMetrics(device=str(self.device))
        
        # 可视化器
        self.visualizer = PointCloudVisualizer()
        
        # TensorBoard
        self.writer = SummaryWriter(config.log_dir)
        
        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=5
        )
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.logger.info("Unsupervised trainer initialized successfully")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单个训练步骤 - 确保内容编码器被训练"""
        self.model.train()
        
        sim_points = batch['sim_points'].to(self.device)
        real_points = batch['real_points'].to(self.device)
        
        batch_size = sim_points.shape[0]
        
        # 随机选择转换方向
        if np.random.rand() > 0.5:
            source_points = sim_points
            target_style = self.model.style_encoder(real_points)
        else:
            source_points = real_points
            target_style = self.model.style_encoder(sim_points)
        
        # 提取内容（从干净的源点云）
        source_content = self.model.content_encoder(source_points)
        
        # 随机时间步
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # 添加噪声
        noise = torch.randn_like(source_points)
        noisy_points = self.diffusion_process.q_sample(source_points, t, noise)
        
        # 从噪声点云提取内容（这会产生梯度）
        noisy_content = self.model.content_encoder(noisy_points)
        
        # 预测噪声 - 使用原始内容
        predicted_noise = self.model(
            noisy_points, t, 
            style_condition=target_style,
            content_condition=source_content  # 使用干净的内容
        )
        
        # 计算去噪后的点云
        with torch.no_grad():
            alpha_t = self.diffusion_process.alphas_cumprod[t][:, None, None]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            x0_pred = (noisy_points - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            generated_points = x0_pred
        
        # 计算损失
        losses = self.criterion(
            pred_noise=predicted_noise,
            target_noise=noise,
            generated_points=generated_points,
            original_points=source_points,
            content_original=source_content,
            content_from_noisy=noisy_content,  # 传入噪声内容
            style_source=self.model.style_encoder(source_points),
            style_target=target_style,
            t=t
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        losses['total'].backward()
        
        # 梯度裁剪
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
        
        self.optimizer.step()
        
        # 更新EMA
        if self.ema is not None:
            self.ema.update()
        
        # 在 train_step 中添加
        if self.global_step % self.config.log_interval == 0:
            # 记录内容编码器的统计信息
            with torch.no_grad():
                content_mean = source_content.mean().item()
                content_std = source_content.std().item()
                content_spatial_var = source_content.var(dim=2).mean().item()
                
            self.writer.add_scalar('debug/content_mean', content_mean, self.global_step)
            self.writer.add_scalar('debug/content_std', content_std, self.global_step)
            self.writer.add_scalar('debug/content_spatial_var', content_spatial_var, self.global_step)
        
        return {k: v.item() for k, v in losses.items()}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证 - 无监督版本"""
        self.logger.info("Running validation...")
        self.model.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                if batch_idx >= 5:  # 只验证前5个batch
                    break
                
                sim_points = batch['sim_points'].to(self.device)
                real_points = batch['real_points'].to(self.device)
                
                # Sim -> Real
                real_style = self.model.style_encoder(real_points)
                sim_content = self.model.content_encoder(sim_points)
                
                sim_to_real = self.diffusion_process.sample(
                    self.model,
                    sim_points.shape,
                    style_condition=real_style,
                    content_condition=sim_content,
                    num_inference_steps=50
                )
                
                # Real -> Sim
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
                metrics = {}
                
                # 内容保持（生成的点云应该保持原始的几何结构）
                content_cd_s2r = self.metrics.chamfer_distance(sim_to_real, sim_points).mean()
                content_cd_r2s = self.metrics.chamfer_distance(real_to_sim, real_points).mean()
                metrics['content_preservation'] = (content_cd_s2r + content_cd_r2s).item() / 2
                
                # 风格转换效果（生成的点云应该接近目标域）
                style_cd_s2r = self.metrics.chamfer_distance(sim_to_real, real_points).mean()
                style_cd_r2s = self.metrics.chamfer_distance(real_to_sim, sim_points).mean()
                metrics['style_transfer'] = (style_cd_s2r + style_cd_r2s).item() / 2
                
                all_metrics.append(metrics)
        
        # 汇总指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[f'val_{key}'] = np.mean(values)
        
        # 综合损失
        avg_metrics['val_loss'] = avg_metrics['val_content_preservation'] + avg_metrics['val_style_transfer']
        
        return avg_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """训练主循环"""
        self.logger.info(f"Starting unsupervised training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            epoch_losses = []
            
            # 训练一个epoch
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
            for batch_idx, batch in enumerate(pbar):
                try:
                    losses = self.train_step(batch)
                    epoch_losses.append(losses)
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': losses.get('total', 0),
                        'diff': losses.get('diffusion', 0),
                        'content': losses.get('content', 0),
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                    
                    # 记录到TensorBoard
                    if self.global_step % self.config.log_interval == 0:
                        for k, v in losses.items():
                            self.writer.add_scalar(f'train/{k}', v, self.global_step)
                        self.writer.add_scalar('train/learning_rate', 
                                             self.optimizer.param_groups[0]['lr'], 
                                             self.global_step)
                    
                    self.global_step += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in training step: {e}")
                    raise
            
            # 计算epoch平均损失
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([l[key] for l in epoch_losses if key in l])
            
            # 记录epoch损失
            for k, v in avg_losses.items():
                self.writer.add_scalar(f'epoch/{k}', v, epoch)
            
            self.logger.info(f"Epoch {epoch} - Average losses:")
            for k, v in avg_losses.items():
                self.logger.info(f"  {k}: {v:.6f}")
            
            # 验证
            if epoch % self.config.eval_interval == 0:
                val_results = self.validate(val_loader)
                
                self.logger.info(f"Validation results:")
                for k, v in val_results.items():
                    self.logger.info(f"  {k}: {v:.6f}")
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                
                # 保存最佳模型
                if val_results['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_results['val_loss']
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best model saved with val_loss: {self.best_val_loss:.6f}")
            
            # 定期保存检查点
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(is_best=False)
            
            # 可视化结果
            if epoch % (self.config.eval_interval * 5) == 0:
                try:
                    self.visualize_results(val_loader, epoch)
                except Exception as e:
                    self.logger.warning(f"Visualization failed: {e}")
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录epoch时间
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        self.logger.info("Unsupervised training completed!")
        self.writer.close()
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict() if self.ema is not None else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        self.checkpoint_manager.save_checkpoint(
            checkpoint,
            epoch=self.current_epoch,
            is_best=is_best,
            metric=self.best_val_loss,
            metric_name='val_loss'
        )
    
    def visualize_results(self, val_loader: DataLoader, epoch: int):
        """可视化训练结果"""
        self.model.eval()
        
        with torch.no_grad():
            # 获取一个batch的数据
            batch = next(iter(val_loader))
            sim_points = batch['sim_points'][:2].to(self.device)  # 只取2个样本
            real_points = batch['real_points'][:2].to(self.device)
            
            # Sim -> Real
            real_style = self.model.style_encoder(real_points)
            sim_content = self.model.content_encoder(sim_points)
            
            sim_to_real = self.diffusion_process.sample(
                self.model,
                sim_points.shape,
                style_condition=real_style,
                content_condition=sim_content,
                num_inference_steps=50
            )
            
            # 创建可视化目录
            vis_dir = os.path.join(self.config.result_dir, f'epoch_{epoch}')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 为每个样本创建可视化
            for i in range(len(sim_points)):
                try:
                    vis_path = os.path.join(vis_dir, f'sample_{i}_sim_to_real.png')
                    self.visualizer.plot_style_transfer_result(
                        sim_points[i].cpu().numpy(),
                        sim_to_real[i].cpu().numpy(),
                        real_points[i].cpu().numpy(),
                        title=f'Epoch {epoch} - Sim to Real - Sample {i+1}',
                        save_path=vis_path
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to visualize sample {i}: {e}")
            
            self.logger.info(f"Visualization saved to: {vis_dir}")
        
        # 恢复训练模式
        self.model.train()