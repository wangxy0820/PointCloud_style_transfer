# training/trainer.py

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm
import os
import math
import numpy as np

from models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
from models.losses import DiffusionLoss
from config.config import Config
from utils.logger import Logger 
from utils.checkpoint import CheckpointManager
from utils.ema import ExponentialMovingAverage

class CosineWithWarmupLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer; self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs; self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr_scale = self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale

class DiffusionTrainer:
    def __init__(self, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device)
        self.logger = Logger(name='DiffusionTrainer', log_dir=config.log_dir, experiment_name=config.experiment_name)
        self.device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        self.model = PointCloudDiffusionModel(config).to(self.device)
        self.diffusion_process = DiffusionProcess(config, device=str(self.device))
        self.loss_fn = DiffusionLoss(noise_weight=1.0, chamfer_weight=config.lambda_chamfer)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95))
        if config.lr_scheduler == "cosine_with_warmup":
            self.scheduler = CosineWithWarmupLR(self.optimizer, config.warmup_epochs, config.num_epochs, config.min_lr_ratio)
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.01)
        self.scaler = GradScaler(enabled=(config.use_amp and self.device_type == 'cuda'))
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=config.ema_decay)
        self.writer = SummaryWriter(log_dir=os.path.join(config.log_dir, config.experiment_name))
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir, config.experiment_name)
        self.best_val_loss = float('inf')
        self.current_epoch = 0; self.patience_counter = 0; self.max_patience = 20
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.gradient_clip_norm = 1.0
        self.logger.info(f"训练器初始化完成:")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info(f"  模型参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        self.logger.info(f"  损失函数: L1 + Chamfer距离")
        self.logger.info(f"  CFG 条件丢弃概率: {self.config.cond_drop_prob}")

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {self.current_epoch}/{self.config.num_epochs} [Train]")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            sim_points = batch['sim_full'].to(self.device)
            real_points = batch['real_full'].to(self.device)
            B, N, C = sim_points.shape
            
            t = torch.randint(0, self.config.num_timesteps, (B,), device=self.device).long()
            noisy_sim_points, actual_noise = self.diffusion_process.q_sample(sim_points, t)
            
            with autocast(device_type=self.device_type, enabled=self.config.use_amp):
                predicted_noise_out, indices = self.model(
                    noisy_points=noisy_sim_points,
                    timestep=t, 
                    condition_points=real_points,
                    cond_drop_prob=self.config.cond_drop_prob,
                    use_hierarchical=self.config.use_hierarchical
                )
                
                loss_dict = {}
                pred_x0_coarse, sim_coarse = None, None
                
                if indices is not None: # Hierarchical path
                    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)
                    actual_noise_coarse = torch.gather(actual_noise, 1, indices_expanded)
                    
                    if self.config.lambda_chamfer > 0:
                        noisy_sim_coarse = torch.gather(noisy_sim_points, 1, indices_expanded)
                        sim_coarse = torch.gather(sim_points, 1, indices_expanded)
                        
                        sqrt_alphas_cumprod_t = self.diffusion_process.sqrt_alphas_cumprod[t].view(B, 1, 1)
                        sqrt_one_minus_alphas_cumprod_t = self.diffusion_process.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
                        
                        pred_x0_coarse = (noisy_sim_coarse - sqrt_one_minus_alphas_cumprod_t * predicted_noise_out) / (sqrt_alphas_cumprod_t + 1e-8)

                    loss, loss_dict = self.loss_fn(
                        predicted_noise=predicted_noise_out,
                        actual_noise=actual_noise_coarse,
                        predicted_points_coarse=pred_x0_coarse,
                        target_points_coarse=sim_coarse
                    )
                else: # Direct path
                    loss, loss_dict = self.loss_fn(
                        predicted_noise=predicted_noise_out,
                        actual_noise=actual_noise
                    )
                
                loss = loss / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(data_loader) - 1:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.ema.update()
            
            total_loss += loss.item() * self.gradient_accumulation_steps

            pbar.set_postfix({
                'Loss': f"{loss_dict.get('total_loss', 0):.4f}",
                'L1': f"{loss_dict.get('noise_loss', 0):.4f}",
                'CD': f"{loss_dict.get('chamfer_loss', 0):.4f}",
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
        avg_loss = total_loss / len(data_loader)
        self.writer.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        return avg_loss

    @torch.no_grad()
    def validate_one_epoch(self, data_loader):
        self.ema.apply_shadow()
        try:
            self.model.eval()
            total_loss = 0.0
            pbar = tqdm(data_loader, desc=f"Epoch {self.current_epoch}/{self.config.num_epochs} [Val]")
            for batch in pbar:
                sim_points = batch['sim_full'].to(self.device)
                real_points = batch['real_full'].to(self.device)
                B, N, C = sim_points.shape
                t = torch.randint(0, self.config.num_timesteps, (B,), device=self.device).long()
                noisy_sim_points, actual_noise = self.diffusion_process.q_sample(sim_points, t)
                
                predicted_noise_out, indices = self.model(
                    noisy_points=noisy_sim_points, timestep=t,
                    condition_points=real_points, cond_drop_prob=0,
                    use_hierarchical=self.config.use_hierarchical)
                
                if indices is not None:
                    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)
                    actual_noise_coarse = torch.gather(actual_noise, 1, indices_expanded)
                    loss, _ = self.loss_fn(predicted_noise_out, actual_noise_coarse)
                else:
                    loss, _ = self.loss_fn(predicted_noise_out, actual_noise)
                
                if torch.isfinite(loss):
                    total_loss += loss.item()
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(data_loader)
            self.writer.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
            return avg_loss
        finally:
            self.ema.restore()

    def save_sample_results(self, data_loader, num_samples: int = 2):
        self.ema.apply_shadow()
        try:
            self.model.eval()
            batch = next(iter(data_loader))
            sim_points = batch['sim_full'][:num_samples].to(self.device)
            real_points = batch['real_full'][:num_samples].to(self.device)
            with torch.no_grad():
                transferred_points = self.diffusion_process.guided_sample_loop(
                    model=self.model, source_points=sim_points,
                    condition_points=real_points, num_inference_steps=50,
                    guidance_scale=self.config.guidance_scale)
            save_dir = os.path.join(self.config.result_dir, self.config.experiment_name, f'epoch_{self.current_epoch:04d}')
            os.makedirs(save_dir, exist_ok=True)
            for i in range(num_samples):
                np.save(os.path.join(save_dir, f'original_sim_{i}.npy'), sim_points[i].cpu().numpy())
                np.save(os.path.join(save_dir, f'reference_real_{i}.npy'), real_points[i].cpu().numpy())
                np.save(os.path.join(save_dir, f'transferred_{i}.npy'), transferred_points[i].cpu().numpy())
            self.logger.info(f"样本结果已保存至 {save_dir}")
        finally:
            self.ema.restore()

    def train(self, train_loader, val_loader):
        self.logger.info("开始最终训练流程...")
        self.current_epoch = self.checkpoint_manager.load(self.model, self.optimizer, self.ema)
        if self.current_epoch > 0:
            self.logger.info(f"从epoch {self.current_epoch}恢复训练")
            for _ in range(self.current_epoch):
                if hasattr(self.scheduler, 'step'): self.scheduler.step()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            avg_train_loss = self.train_one_epoch(train_loader)
            self.logger.info(f"Epoch {epoch}: 训练损失: {avg_train_loss:.6f}")
            if hasattr(self.scheduler, 'step'): self.scheduler.step()

            if epoch % self.config.val_interval == 0:
                avg_val_loss = self.validate_one_epoch(val_loader)
                self.logger.info(f"Epoch {epoch}: 验证损失: {avg_val_loss:.6f}")
                is_best = avg_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_val_loss; self.patience_counter = 0
                    self.logger.info(f"新的最佳模型! 验证损失: {self.best_val_loss:.6f}")
                else:
                    self.patience_counter += 1
                
                self.checkpoint_manager.save(self.model, self.optimizer, self.ema, epoch, is_best)
                
                if self.patience_counter >= self.max_patience:
                    self.logger.info(f"验证损失{self.patience_counter}个epochs无改善，早停")
                    break
                
                if epoch > 0 and epoch % (self.config.save_interval * 2) == 0:
                    self.save_sample_results(val_loader)
        
        self.logger.info(f"训练完成! 最佳验证损失: {self.best_val_loss:.6f}")
        self.writer.close()