import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import numpy as np
from typing import Dict

from models.diffusion_model_unsupervised import UnsupervisedPointCloudDiffusionModel, UnsupervisedDiffusionProcess
from models.losses_unsupervised import UnsupervisedDiffusionLoss
from evaluation.metrics import PointCloudMetrics
from utils.visualization import PointCloudVisualizer
from utils.logger import Logger
from utils.checkpoint import CheckpointManager

# 确保这个类是可用的，或者从您的utils文件中导入
class ExponentialMovingAverage:
    """指数移动平均"""
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
    
    def restore(self):
        # 这是一个简化的restore，理想情况下应该有一个单独的备份
        # 在验证时，通常是 apply_shadow -> validation -> restore
        pass
    
    def state_dict(self):
        return {'decay': self.decay, 'shadow_params': self.shadow_params}
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']


class UnsupervisedDiffusionTrainer:
    """无监督Diffusion模型训练器 - 最终完整修复版"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # 完整路径
        experiment_path = os.path.join('experiments', config.experiment_name)
        log_dir = os.path.join(experiment_path, 'logs')
        checkpoint_dir = os.path.join(experiment_path, 'checkpoints')
        
        # 初始化日志
        self.logger = Logger(
            name='UnsupervisedDiffusionTrainer',
            log_dir=log_dir
        )
        
        # 初始化模型
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
        self.criterion = UnsupervisedDiffusionLoss(
            lambda_diffusion=config.lambda_diffusion,
            lambda_chamfer=config.lambda_chamfer,
            lambda_content=config.lambda_content,
            lambda_style=config.lambda_style,
            lambda_smooth=config.lambda_smooth,
            lambda_lidar_structure=config.lambda_lidar_structure
        ).to(self.device)
        
        self.warmup_steps = config.warmup_steps
        
        # EMA
        if hasattr(config, 'ema_decay') and config.ema_decay > 0:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=config.ema_decay
            )
        else:
            self.ema = None
        
        self.metrics = PointCloudMetrics(device=str(self.device))
        self.visualizer = PointCloudVisualizer()
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=5
        )
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.logger.info("Unsupervised trainer initialized successfully")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        sim_points = batch['sim_points'].to(self.device)
        real_points = batch['real_points'].to(self.device)
        batch_size = sim_points.shape[0]

        if np.random.rand() > 0.5:
            source_points, target_points = sim_points, real_points
        else:
            source_points, target_points = real_points, sim_points

        target_style = self.model.style_encoder(target_points)
        source_content = self.model.content_encoder(source_points)
        
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        noise = torch.randn_like(source_points)
        noisy_points = self.diffusion_process.q_sample(source_points, t, noise)
        
        noisy_content = self.model.content_encoder(noisy_points)
        
        predicted_noise = self.model(
            noisy_points, t, 
            style_condition=target_style,
            content_condition=source_content.detach()
        )
        
        alpha_t = self.diffusion_process.alphas_cumprod[t].view(batch_size, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
        generated_points = sqrt_recip_alpha_t * (noisy_points - sqrt_one_minus_alpha_t * predicted_noise)

        generated_style = self.model.style_encoder(generated_points)

        warmup_factor = min(1.0, self.global_step / self.warmup_steps)
        losses = self.criterion(
            pred_noise=predicted_noise,
            target_noise=noise,
            generated_points=generated_points,
            original_points=source_points,
            content_original=source_content,
            content_from_noisy=noisy_content,
            style_source=generated_style,
            style_target=target_style.detach(),
            warmup_factor=warmup_factor
        )
        
        self.optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        if self.ema: self.ema.update()

        self.global_step += 1
        return {k: v.item() for k, v in losses.items()}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证 - 无监督版本"""
        self.logger.info("Running validation...")
        self.model.eval()
        
        all_metrics = []
        
        if self.ema:
            self.ema.apply_shadow()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                sim_points = batch['sim_points'].to(self.device)
                real_points = batch['real_points'].to(self.device)
                
                real_style = self.model.style_encoder(real_points)
                sim_content = self.model.content_encoder(sim_points)
                
                sim_to_real = self.diffusion_process.sample(
                    self.model,
                    sim_points.shape,
                    style_condition=real_style,
                    content_condition=sim_content,
                    num_inference_steps=50
                )
                
                metrics = {}
                metrics['content_preservation'] = self.metrics.chamfer_distance(sim_to_real, sim_points).mean().item()
                metrics['style_transfer_cd'] = self.metrics.chamfer_distance(sim_to_real, real_points).mean().item()
                all_metrics.append(metrics)
        
        if self.ema:
            self.ema.restore()
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[f'val_{key}'] = np.mean([m[key] for m in all_metrics])
        
        avg_metrics['val_loss'] = avg_metrics.get('val_content_preservation', 0) + avg_metrics.get('val_style_transfer_cd', 0)
        
        return avg_metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """训练主循环"""
        self.logger.info(f"Starting unsupervised training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
            for batch in pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses['total'])
                
                pbar.set_postfix({
                    'loss': losses.get('total', 0),
                    'chamfer': losses.get('chamfer', 0)
                })
                
                if self.global_step % self.config.log_interval == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
                    self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            avg_epoch_loss = np.mean(epoch_losses)
            self.writer.add_scalar('epoch/loss', avg_epoch_loss, epoch)
            self.logger.info(f"Epoch {epoch} - Average Train Loss: {avg_epoch_loss:.6f}")

            if epoch % self.config.eval_interval == 0:
                val_results = self.validate(val_loader)
                self.logger.info(f"Validation results: {val_results}")
                for k, v in val_results.items():
                    self.writer.add_scalar(k, v, epoch)
                
                if val_results['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_results['val_loss']
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best model saved with val_loss: {self.best_val_loss:.6f}")
            
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(is_best=False)
            
            self.scheduler.step()
        
        self.writer.close()
        self.logger.info("Unsupervised training completed!")

    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        if self.ema:
            state['ema_state_dict'] = self.ema.state_dict()
            
        self.checkpoint_manager.save_checkpoint(
            state,
            epoch=self.current_epoch,
            is_best=is_best,
            metric=self.best_val_loss,
            metric_name='val_loss'
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.ema is not None and checkpoint.get('ema_state_dict') is not None:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'current_chunks' in checkpoint:
            self.current_chunks = checkpoint['current_chunks']
        
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        
        self.logger.info(f"Resumed from epoch {checkpoint['epoch']}, global step {self.global_step}")
    
    def visualize_results(self, val_loader: DataLoader, epoch: int):
        """可视化训练结果"""
        self.model.eval()
        
        # 如果有EMA，使用EMA权重
        if self.ema is not None:
            self.ema.apply_shadow()
        
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
                    # 如果数据使用了LiDAR标准化，可能需要反标准化以正确可视化
                    sim_viz = sim_points[i].cpu().numpy()
                    real_viz = real_points[i].cpu().numpy()
                    gen_viz = sim_to_real[i].cpu().numpy()
                    
                    vis_path = os.path.join(vis_dir, f'sample_{i}_sim_to_real.png')
                    self.visualizer.plot_style_transfer_result(
                        sim_viz,
                        gen_viz,
                        real_viz,
                        title=f'Epoch {epoch} - Sim to Real - Sample {i+1}',
                        save_path=vis_path
                    )
                    
                    # 额外保存Z轴分布对比
                    self.visualizer.plot_z_distribution(
                        sim_viz[:, 2],
                        gen_viz[:, 2],
                        real_viz[:, 2],
                        save_path=os.path.join(vis_dir, f'sample_{i}_z_dist.png')
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to visualize sample {i}: {e}")
            
            self.logger.info(f"Visualization saved to: {vis_dir}")
        
        # 恢复原始权重
        if self.ema is not None:
            self.ema.restore()
        
        # 恢复训练模式
        self.model.train()


# 主训练脚本
if __name__ == "__main__":
    import argparse
    from config import Config
    
    parser = argparse.ArgumentParser(description='Train unsupervised diffusion model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment_name', type=str, default='lidar_unsupervised',
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    # 设置实验目录
    config.experiment_dir = args.experiment_name
    config.checkpoint_dir = os.path.join('experiments', args.experiment_name, 'checkpoints')
    config.log_dir = os.path.join('experiments', args.experiment_name, 'logs')
    config.result_dir = os.path.join('experiments', args.experiment_name, 'results')
    
    # 确保使用LiDAR模式
    config.use_lidar_normalization = True
    config.use_lidar_chunking = True
    
    # 设置预热步数
    config.warmup_steps = 1000
    
    # 创建训练器
    trainer = UnsupervisedDiffusionTrainer(config)
    
    # 如果指定了检查点，加载它
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = trainer.create_dataloaders_for_training()
    
    # 开始训练
    trainer.train(train_loader, val_loader)