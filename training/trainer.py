# training/trainer.py

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os
import math

from models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
from models.losses import DiffusionLoss
from config.config import Config
from utils.logger import Logger 
from utils.checkpoint import CheckpointManager
from utils.ema import ExponentialMovingAverage

class CosineWithWarmupLR:
    """带warmup的余弦学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
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
        self.logger = Logger(name='HierarchicalTrainer', log_dir=config.log_dir, experiment_name=config.experiment_name)
        
        self.device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        
        self.model = PointCloudDiffusionModel(config).to(self.device)
        self.diffusion_process = DiffusionProcess(config, device=str(self.device))
        self.loss_fn = DiffusionLoss(loss_type='l1')

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        if config.lr_scheduler == "cosine_with_warmup":
            self.scheduler = CosineWithWarmupLR(
                self.optimizer, 
                config.warmup_epochs, 
                config.num_epochs, 
                config.min_lr_ratio
            )
        else: # Fallback to a standard scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.01
            )
        
        self.scaler = GradScaler(enabled=(config.use_amp and self.device_type == 'cuda'))
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=config.ema_decay)
        self.writer = SummaryWriter(log_dir=os.path.join(config.log_dir, config.experiment_name))
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir, config.experiment_name)

        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        self.logger.info(f"Trainer initialized with device: {self.device} (type: {self.device_type})")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        self.logger.info(f"Using scheduler: {config.lr_scheduler}")

    def train_one_epoch(self, data_loader):
        self.model.train()
        pbar = tqdm(data_loader, desc=f"Epoch {self.current_epoch}/{self.config.num_epochs} [Train]")
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            # 直接从batch获取完整的点云数据
            sim_points = batch['sim_full'].to(self.device)
            real_points = batch['real_full'].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            B = sim_points.shape[0]

            # 总是使用分层处理
            use_hierarchical = self.config.use_hierarchical

            # 对simulation点云加噪，用real点云作为条件
            t = torch.randint(0, self.config.num_timesteps, (B,), device=self.device).long()
            noisy_sim_points, actual_noise = self.diffusion_process.q_sample(x_start=sim_points, t=t)
            
            with autocast(device_type=self.device_type, enabled=self.config.use_amp):
                predicted_noise = self.model(noisy_sim_points, t, real_points, use_hierarchical=use_hierarchical)
                loss = self.loss_fn(predicted_noise, actual_noise)
            
            if not torch.isfinite(loss):
                self.logger.warning(f"Non-finite loss detected: {loss.item()}, skipping batch")
                continue
            
            self.scaler.scale(loss).backward()
            
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update()
            
            if batch_idx % 10 == 0:
                    noise_std = predicted_noise.std()
                    # 将多样性损失的权重从 0.05 提高到 0.1
                    diversity_loss = torch.relu(0.1 - noise_std) * 0.1
                    loss = loss + diversity_loss
            
            loss_item = loss.item()
            total_loss += loss_item
            
            pbar.set_postfix({
                'loss': f"{loss_item:.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        return total_loss / len(data_loader)
    
    @torch.no_grad()
    def validate_one_epoch(self, data_loader):
        self.ema.apply_shadow()
        try:
            self.model.eval()
            pbar = tqdm(data_loader, desc=f"Epoch {self.current_epoch}/{self.config.num_epochs} [Val]")
            total_loss = 0.0
            
            for batch in pbar:
                # 直接从batch获取完整的点云数据
                sim_points = batch['sim_full'].to(self.device)
                real_points = batch['real_full'].to(self.device)
                B = sim_points.shape[0]

                t = torch.randint(0, self.config.num_timesteps, (B,), device=self.device).long()
                noisy_sim_points, actual_noise = self.diffusion_process.q_sample(x_start=sim_points, t=t)
                
                with autocast(device_type=self.device_type, enabled=self.config.use_amp):
                    predicted_noise = self.model(noisy_sim_points, t, real_points, use_hierarchical=self.config.use_hierarchical)
                    loss = self.loss_fn(predicted_noise, actual_noise)
                    
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    
            avg_loss = total_loss / len(data_loader)
            self.writer.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
            return avg_loss
        finally:
            self.ema.restore()

    def train(self, train_loader, val_loader):
        self.logger.info("Starting hierarchical training process...")
        
        self.current_epoch = self.checkpoint_manager.load(self.model, self.optimizer, self.ema)
        if self.current_epoch > 0:
            self.logger.info(f"Resumed from epoch {self.current_epoch}")
            for _ in range(self.current_epoch):
                self.scheduler.step()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            avg_train_loss = self.train_one_epoch(train_loader)
            self.logger.info(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.4f}")
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            if epoch % self.config.val_interval == 0:
                avg_val_loss = self.validate_one_epoch(val_loader)
                self.logger.info(f"Epoch {epoch}: Avg Validation Loss: {avg_val_loss:.4f}")
                
                is_best = avg_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_val_loss
                    self.logger.info(f"New best model with validation loss: {self.best_val_loss:.4f}")
                
                self.checkpoint_manager.save(self.model, self.optimizer, self.ema, epoch, is_best)

            if epoch % self.config.save_interval == 0 and epoch > 0:
                self.checkpoint_manager.save(self.model, self.optimizer, self.ema, epoch)

            self.scheduler.step()
            
        self.logger.info("Training completed.")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")