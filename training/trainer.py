"""
Diffusion模型训练器
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

from models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
from models.pointnet2_encoder import ImprovedPointNet2Encoder
from models.losses import DiffusionLoss
from evaluation.metrics import PointCloudMetrics
from utils.visualization import PointCloudVisualizer
from utils.logger import Logger
from utils.checkpoint import CheckpointManager
from training.validator import Validator


class ExponentialMovingAverage:
    """指数移动平均"""
    
    def __init__(self, parameters, decay: float = 0.995):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in self.parameters]
    
    def update(self):
        for param, shadow_param in zip(self.parameters, self.shadow_params):
            shadow_param.data = self.decay * shadow_param.data + (1 - self.decay) * param.data
    
    def apply(self):
        for param, shadow_param in zip(self.parameters, self.shadow_params):
            param.data = shadow_param.data
    
    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow_params': self.shadow_params
        }
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']


class DiffusionTrainer:
    """Diffusion模型训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # 初始化日志
        self.logger = Logger(
            name='DiffusionTrainer',
            log_dir=config.log_dir
        )
        
        # 初始化模型
        self.logger.info("Initializing models...")
        self.model = PointCloudDiffusionModel(
            input_dim=3,
            hidden_dims=[128, 256, 512, 1024],
            time_dim=config.time_embed_dim
        ).to(self.device)
        
        # 风格编码器 - 确保输出维度与模型期望的匹配
        self.style_encoder = ImprovedPointNet2Encoder(
            input_channels=3,
            feature_dim=1024  # 这必须与DiffusionModel中的hidden_dims[-1]匹配
        ).to(self.device)
        
        # Diffusion过程
        self.diffusion_process = DiffusionProcess(
            num_timesteps=config.num_timesteps,
            beta_schedule=config.beta_schedule,
            device=self.device
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.style_encoder.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # 损失函数
        self.criterion = DiffusionLoss(
            lambda_reconstruction=config.lambda_reconstruction,
            lambda_perceptual=config.lambda_perceptual,
            lambda_continuity=config.lambda_continuity,
            lambda_boundary=config.lambda_boundary
        )
        
        # EMA
        if config.ema_decay > 0:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=config.ema_decay
            )
        else:
            self.ema = None
        
        # 评估器
        self.metrics = PointCloudMetrics(device=str(self.device))
        self.validator = Validator(device=str(self.device))
        
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
        
        # 评估间隔
        self.config.eval_interval = getattr(config, 'eval_interval', 1)
        
        self.logger.info("Trainer initialized successfully")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单个训练步骤"""
        self.model.train()
        self.style_encoder.train()
        
        sim_points = batch['sim_points'].to(self.device)
        real_points = batch['real_points'].to(self.device)
        
        batch_size = sim_points.shape[0]
        
        # 提取风格特征
        # 风格编码器输出: [B, feature_dim]
        style_features = self.style_encoder(real_points)  # [B, 1024]
        
        # 为交叉注意力机制添加序列维度
        # 从 [B, feature_dim] -> [B, 1, feature_dim]
        style_features = style_features.unsqueeze(1)  # [B, 1, 1024]
        
        # 随机时间步
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # 添加噪声
        noise = torch.randn_like(sim_points)
        noisy_points = self.diffusion_process.q_sample(sim_points, t, noise)
        
        # 预测噪声
        predicted_noise = self.model(noisy_points, t, style_features)
        
        # 计算损失
        losses = self.criterion(
            predicted_noise, 
            noise,
            predicted_features=None,
            target_features=None,
            chunks=None
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        losses['total'].backward()
        
        # 梯度裁剪
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.style_encoder.parameters()),
                self.config.gradient_clip
            )
        
        self.optimizer.step()
        
        # 更新EMA
        if self.ema is not None:
            self.ema.update()
        
        return {k: v.item() for k, v in losses.items()}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.logger.info("Running validation...")
        
        # 使用validator进行验证
        val_results = self.validator.validate(
            self.model,
            self.style_encoder,
            self.diffusion_process,
            val_loader,
            num_inference_steps=50  # 快速验证
        )
        
        return val_results
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """训练主循环"""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
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
                        'loss': losses['total'],
                        'recon': losses.get('reconstruction', 0),
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
                    self.logger.error(f"Batch shapes - sim: {batch['sim_points'].shape}, real: {batch['real_points'].shape}")
                    raise
            
            # 计算epoch平均损失
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([l[key] for l in epoch_losses])
            
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
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'style_encoder_state_dict': self.style_encoder.state_dict(),
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
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.style_encoder.load_state_dict(checkpoint['style_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.ema is not None and checkpoint['ema_state_dict'] is not None:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def visualize_results(self, val_loader: DataLoader, epoch: int):
        """可视化训练结果"""
        self.model.eval()
        self.style_encoder.eval()
        
        with torch.no_grad():
            # 获取一个batch的数据
            batch = next(iter(val_loader))
            sim_points = batch['sim_points'][:4].to(self.device)  # 只取4个样本
            real_points = batch['real_points'][:4].to(self.device)
            
            # 生成
            style_features = self.style_encoder(real_points).unsqueeze(1)
            
            # 快速采样用于可视化
            generated = self.diffusion_process.sample(
                self.model,
                sim_points.shape,
                style_features
            )
            
            # 创建可视化目录
            vis_dir = os.path.join(self.config.result_dir, f'epoch_{epoch}')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 为每个样本创建可视化
            for i in range(len(sim_points)):
                try:
                    vis_path = os.path.join(vis_dir, f'sample_{i}.png')
                    self.visualizer.plot_style_transfer_result(
                        sim_points[i].cpu().numpy(),
                        generated[i].cpu().numpy(),
                        real_points[i].cpu().numpy(),
                        title=f'Epoch {epoch} - Sample {i+1}',
                        save_path=vis_path
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to visualize sample {i}: {e}")
            
            self.logger.info(f"Visualization saved to: {vis_dir}")
        
        # 恢复训练模式
        self.model.train()
        self.style_encoder.train()