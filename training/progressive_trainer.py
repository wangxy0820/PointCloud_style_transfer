"""
渐进式训练器
"""

import torch
import torch.nn as nn
from typing import Dict, List
from tqdm import tqdm

from training.trainer import DiffusionTrainer


class ProgressiveDiffusionTrainer(DiffusionTrainer):
    """渐进式Diffusion训练器"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # 渐进式训练参数
        self.current_chunks = config.initial_chunks
        self.max_chunks = config.num_chunks_per_pc
        self.chunks_increment = config.chunks_increment
        self.progressive_epochs = config.progressive_epochs
        
        # 当前阶段
        self.current_stage = 0
        self.stage_start_epoch = 0
    
    def should_increase_chunks(self) -> bool:
        """判断是否应该增加块数"""
        epochs_in_stage = self.current_epoch - self.stage_start_epoch
        return (epochs_in_stage >= self.progressive_epochs and 
                self.current_chunks < self.max_chunks)
    
    def increase_chunks(self):
        """增加训练块数"""
        self.current_chunks = min(
            self.current_chunks + self.chunks_increment,
            self.max_chunks
        )
        self.current_stage += 1
        self.stage_start_epoch = self.current_epoch
        
        print(f"\n{'='*60}")
        print(f"Progressive Training: Stage {self.current_stage}")
        print(f"Increasing chunks to: {self.current_chunks}/{self.max_chunks}")
        print(f"{'='*60}\n")
        
        # 调整学习率
        self.adjust_learning_rate_for_stage()
    
    def adjust_learning_rate_for_stage(self):
        """为新阶段调整学习率"""
        # 可以选择重置学习率或继续衰减
        # 这里选择轻微提升学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 1.5  # 提升50%
    
    def create_progressive_dataloader(self, full_loader):
        """创建渐进式数据加载器"""
        # 这里需要修改数据加载逻辑，只使用部分块
        # 简化示例：直接返回原始loader
        # 实际实现中应该过滤数据集，只使用current_chunks个块
        return full_loader
    
    def train(self, train_loader, val_loader):
        """渐进式训练主循环"""
        print(f"Starting progressive training with initial {self.current_chunks} chunks")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # 检查是否需要增加块数
            if self.should_increase_chunks():
                self.increase_chunks()
                # 可以选择重新创建数据加载器
                # train_loader = self.create_progressive_dataloader(full_train_loader)
            
            # 显示当前阶段信息
            print(f"\nEpoch {epoch} - Stage {self.current_stage} - Chunks: {self.current_chunks}")
            
            # 正常训练流程
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
            for batch in pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                
                pbar.set_postfix({
                    'loss': losses['total'],
                    'chunks': self.current_chunks
                })
                
                if self.global_step % 50 == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
                    self.writer.add_scalar('train/current_chunks', self.current_chunks, self.global_step)
                
                self.global_step += 1
            
            # 验证
            val_results = self.validate(val_loader)
            print(f"Validation - Loss: {val_results['val_loss']:.6f}, "
                  f"Chamfer Distance: {val_results['val_chamfer_distance']:.6f}")
            
            # 保存检查点
            if val_results['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_results['val_loss']
                self.save_checkpoint(is_best=True)
            
            if epoch % 10 == 0:
                self.save_checkpoint(is_best=False)
            
            self.scheduler.step()
        
        print("Progressive training completed!")
