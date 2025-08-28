import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import numpy as np
from typing import Dict

from models.diffusion_model_unsupervised import UnsupervisedPointCloudDiffusionModel, UnsupervisedDiffusionProcess
from models.losses_unsupervised import UnsupervisedDiffusionLoss
from evaluation.metrics import PointCloudMetrics
from utils.visualization import PointCloudVisualizer
from utils.logger import Logger
from utils.checkpoint import CheckpointManager


class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.995):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in self.parameters]
        
    def update(self):
        for param, shadow_param in zip(self.parameters, self.shadow_params):
            shadow_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        for param, shadow_param in zip(self.parameters, self.shadow_params):
            param.data.copy_(shadow_param.data)
    
    def restore(self, model_params):
        for param, shadow_param in zip(model_params, self.shadow_params):
            param.data.copy_(shadow_param.data)

    def state_dict(self):
        return {'decay': self.decay, 'shadow_params': self.shadow_params}
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']


class UnsupervisedDiffusionTrainer:
    """
    修复的训练器 - 解决验证loss过高的问题
    """
    def __init__(self, config, stage: int = 1, stage1_checkpoint_path: str = None):
        self.config = config
        self.stage = stage
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        experiment_path = os.path.join('experiments', config.experiment_name)
        log_dir = os.path.join(experiment_path, 'logs')
        checkpoint_dir = os.path.join(experiment_path, 'checkpoints')
        
        self.logger = Logger(name=f'UnsupervisedTrainer_Stage{self.stage}', log_dir=log_dir)
        self.writer = SummaryWriter(log_dir)
        
        # 创建模型
        self.model = UnsupervisedPointCloudDiffusionModel(
            input_dim=3,
            hidden_dims=[128, 256, 512, 1024],
            time_dim=config.time_embed_dim,
            style_dim=256,
            content_dims=[64, 128, 256]
        ).to(self.device)
        
        # Stage 2: 加载Stage 1的权重
        if self.stage == 2:
            if not stage1_checkpoint_path or not os.path.exists(stage1_checkpoint_path):
                raise ValueError("Stage 2 requires Stage 1 checkpoint")
            self.logger.info(f"Loading Stage 1 checkpoint: {stage1_checkpoint_path}")
            checkpoint = torch.load(stage1_checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 冻结部分层
            self.logger.info("Freezing Content Encoder for Stage 2")
            for param in self.model.content_encoder.parameters():
                param.requires_grad = False
        
        # Diffusion过程
        self.diffusion_process = UnsupervisedDiffusionProcess(
            num_timesteps=config.num_timesteps,
            beta_schedule=config.beta_schedule,
            device=self.device
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs
        )
        
        # 损失函数 - Stage 1使用更简单的配置
        if self.stage == 1:
            loss_kwargs = {
                'lambda_diffusion': 1.0,
                'lambda_chamfer': 0.0,  # 初始不使用chamfer loss
                'lambda_content': 0.1,
                'lambda_style': 0.0,
                'lambda_smooth': 0.01,
                'lambda_lidar_structure': 0.0
            }
        else:
            loss_kwargs = {
                'lambda_diffusion': 1.0,
                'lambda_chamfer': 2.0,
                'lambda_content': 0.1,
                'lambda_style': 0.1,
                'lambda_smooth': 0.01,
                'lambda_lidar_structure': 0.1
            }
        
        self.criterion = UnsupervisedDiffusionLoss(**loss_kwargs).to(self.device)
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # EMA
        if config.ema_decay > 0:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=config.ema_decay)
        else:
            self.ema = None
        
        # 评估
        self.metrics = PointCloudMetrics(device=str(self.device))
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir, max_checkpoints=5)
        
        # 状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.logger.info(f"Trainer initialized for Stage {self.stage}")
        self.logger.info(f"Loss weights: {loss_kwargs}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        self.model.train()
        self.optimizer.zero_grad()
        
        sim_points = batch['sim_points'].to(self.device)
        real_points = batch['real_points'].to(self.device)
        
        # 验证输入数据范围
        sim_min, sim_max = sim_points.min().item(), sim_points.max().item()
        real_min, real_max = real_points.min().item(), real_points.max().item()
        
        if abs(sim_min) > 2 or abs(sim_max) > 2 or abs(real_min) > 2 or abs(real_max) > 2:
            self.logger.warning(f"Input data out of range: sim[{sim_min:.2f}, {sim_max:.2f}], real[{real_min:.2f}, {real_max:.2f}]")
            # Clamp到合理范围
            sim_points = torch.clamp(sim_points, -1.5, 1.5)
            real_points = torch.clamp(real_points, -1.5, 1.5)
        
        with autocast():
            batch_size = sim_points.shape[0]
            
            # Stage 1: 自重建
            if self.stage == 1:
                source_points = sim_points
                target_points = sim_points  # 自重建
                target_style = self.model.style_encoder(source_points)
            else:
                # Stage 2: 风格转换
                if np.random.rand() > 0.5:
                    source_points = sim_points
                    target_points = real_points
                else:
                    source_points = real_points
                    target_points = sim_points
                target_style = self.model.style_encoder(target_points)
            
            # 内容编码
            source_content = self.model.content_encoder(source_points)
            
            # Diffusion前向过程 - 使用较小的时间步
            max_t = min(500, self.config.num_timesteps)
            t = torch.randint(0, max_t, (batch_size,), device=self.device)
            
            # 生成噪声
            noise = torch.randn_like(source_points)
            
            # 加噪
            noisy_points = self.diffusion_process.q_sample(source_points, t, noise)
            
            # 限制noisy_points范围
            noisy_points = torch.clamp(noisy_points, -2.0, 2.0)
            
            # 预测噪声
            if self.stage == 1:
                # Stage 1: 不使用风格条件
                predicted_noise = self.model(
                    noisy_points, t,
                    style_condition=None,
                    content_condition=source_content
                )
            else:
                # Stage 2: 使用风格条件
                predicted_noise = self.model(
                    noisy_points, t,
                    style_condition=target_style,
                    content_condition=source_content
                )
            
            # 计算重建（用于辅助损失）
            alpha_t = self.diffusion_process.alphas_cumprod[t].view(batch_size, 1, 1)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
            
            generated_points = sqrt_recip_alpha_t * (noisy_points - sqrt_one_minus_alpha_t * predicted_noise)
            generated_points = torch.clamp(generated_points, -1.5, 1.5)
            
            # 提取内容和风格
            noisy_content = self.model.content_encoder(noisy_points)
            generated_style = self.model.style_encoder(generated_points)
            
            # 计算损失
            warmup_factor = min(1.0, self.global_step / self.config.warmup_steps)
            
            # 动态调整chamfer loss权重
            if self.current_epoch < 10:
                chamfer_weight = 0.0  # 前10个epoch不使用
            elif self.current_epoch < 30:
                chamfer_weight = ((self.current_epoch - 10) / 20) * 2.0  # 线性增加
            else:
                chamfer_weight = 2.0 if self.stage == 1 else 5.0
            
            # 临时修改损失权重
            original_lambda_chamfer = self.criterion.lambda_chamfer
            self.criterion.lambda_chamfer = chamfer_weight
            
            losses = self.criterion(
                pred_noise=predicted_noise,
                target_noise=noise,
                generated_points=generated_points,
                original_points=target_points,
                content_original=source_content,
                content_from_noisy=noisy_content,
                style_source=generated_style,
                style_target=target_style.detach() if target_style is not None else generated_style,
                warmup_factor=warmup_factor,
                epoch=self.current_epoch
            )
            
            # 恢复原始权重
            self.criterion.lambda_chamfer = original_lambda_chamfer
            
            total_loss = losses['total']
        
        # 反向传播
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        # 更新权重
        if grad_norm < 10:  # 只在梯度正常时更新
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.ema:
                self.ema.update()
        else:
            self.logger.warning(f"Gradient explosion: {grad_norm:.2f}, skipping update")
            self.scaler.update()
        
        self.global_step += 1
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证 - 使用更快的推理"""
        self.model.eval()
        all_metrics = []
        
        if self.ema:
            self.ema.apply_shadow()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                if i >= 5:  # 只验证5个batch以加快速度
                    break
                
                sim_points = batch['sim_points'].to(self.device)
                real_points = batch['real_points'].to(self.device)
                
                # Clamp输入
                sim_points = torch.clamp(sim_points, -1.5, 1.5)
                real_points = torch.clamp(real_points, -1.5, 1.5)
                
                if self.stage == 1:
                    source_points = sim_points
                    target_points = sim_points
                    target_style = None
                else:
                    source_points = sim_points
                    target_points = real_points
                    target_style = self.model.style_encoder(target_points)
                
                content_features = self.model.content_encoder(source_points)
                
                # 快速生成（使用更少的步数）
                generated = self.diffusion_process.sample(
                    self.model,
                    source_points.shape,
                    style_condition=target_style,
                    content_condition=content_features,
                    num_inference_steps=10  # 减少到10步
                )
                
                # 计算指标
                metrics = {}
                
                # 内容保持
                cd = self.metrics.chamfer_distance(generated, source_points)
                metrics['content_preservation'] = cd.mean().item()
                
                if self.stage == 2:
                    # 风格转换
                    cd_style = self.metrics.chamfer_distance(generated, target_points)
                    metrics['style_transfer_cd'] = cd_style.mean().item()
                
                all_metrics.append(metrics)
        
        if self.ema:
            self.ema.restore(self.model.parameters())
        
        # 计算平均
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[f'val_{key}'] = np.mean([m[key] for m in all_metrics])
        
        # 总验证损失
        if self.stage == 1:
            avg_metrics['val_loss'] = avg_metrics.get('val_content_preservation', float('inf'))
        else:
            avg_metrics['val_loss'] = avg_metrics.get('val_style_transfer_cd', float('inf'))
        
        return avg_metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """主训练循环"""
        self.logger.info(f"Starting Stage {self.stage} training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f'Stage {self.stage} - Epoch {epoch}/{self.config.num_epochs}')
            for batch in pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses['total'])
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{losses['total']:.3f}",
                    'diff': f"{losses.get('diffusion', 0):.3f}",
                    'chamfer': f"{losses.get('chamfer', 0):.3f}"
                })
                
                # 记录到tensorboard
                if self.global_step % self.config.log_interval == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            # Epoch统计
            avg_epoch_loss = np.mean(epoch_losses)
            self.writer.add_scalar('epoch/loss', avg_epoch_loss, epoch)
            self.logger.info(f"Epoch {epoch} - Average Loss: {avg_epoch_loss:.4f}")
            
            # 验证
            if epoch % self.config.eval_interval == 0:
                val_results = self.validate(val_loader)
                self.logger.info(f"Validation: {val_results}")
                
                for k, v in val_results.items():
                    self.writer.add_scalar(k, v, epoch)
                
                # 保存最佳模型
                if val_results['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_results['val_loss']
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best model! Val loss: {self.best_val_loss:.4f}")
            
            # 定期保存
            if epoch % self.config.save_interval == 0 and epoch > 0:
                self.save_checkpoint(is_best=False)
            
            # 更新学习率
            self.scheduler.step()
        
        self.writer.close()
        self.logger.info(f"Stage {self.stage} training completed!")

    def save_checkpoint(self, is_best: bool = False):
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'stage': self.stage
        }
        if self.ema:
            state['ema_state_dict'] = self.ema.state_dict()
        
        self.checkpoint_manager.save_checkpoint(
            state, epoch=self.current_epoch, is_best=is_best,
            metric=self.best_val_loss, metric_name='val_loss'
        )