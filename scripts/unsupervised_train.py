#!/usr/bin/env python3
"""
无监督训练脚本
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
from datetime import datetime
import logging

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.dataset import create_dataloaders
from training.unsupervised_trainer import UnsupervisedDiffusionTrainer
from utils.logger import Logger
from utils.checkpoint import CheckpointManager


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train Unsupervised Point Cloud Style Transfer')
    
    # 必需参数
    parser.add_argument('--data_dir', type=str, required=True, help='Preprocessed data directory')
    
    # 实验配置
    parser.add_argument('--experiment_name', type=str, default='unsupervised_diffusion')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 其他选项
    parser.add_argument('--use_ema', action='store_true', help='Use exponential moving average')
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=2)
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建配置
    config = Config()
    
    # 覆盖配置参数
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    
    # 设置其他参数
    config.device = args.device
    config.num_workers = args.num_workers
    config.gradient_clip = args.gradient_clip
    config.save_interval = args.save_interval
    config.log_interval = args.log_interval
    config.eval_interval = args.eval_interval
    
    # 如果使用EMA
    if args.use_ema:
        config.ema_decay = 0.995
    else:
        config.ema_decay = 0
    
    # 更新路径
    experiment_dir = os.path.join('experiments', args.experiment_name)
    config.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    config.log_dir = os.path.join(experiment_dir, 'logs')
    config.result_dir = os.path.join(experiment_dir, 'results')
    
    # 创建目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # 初始化日志
    logger = Logger(
        name='unsupervised_training',
        log_dir=config.log_dir,
        log_level='INFO'
    )
    
    logger.info(f"Starting unsupervised experiment: {args.experiment_name}")
    logger.info(f"Configuration: {config.__dict__}")
    
    # 保存配置
    import json
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir,
            config.batch_size,
            config.num_workers,
            config.chunk_size
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # 创建训练器
    logger.info("Creating unsupervised trainer...")
    try:
        trainer = UnsupervisedDiffusionTrainer(config)
        
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return
    
    # 恢复训练
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=config.device, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'ema_state_dict' in checkpoint and args.use_ema:
                trainer.ema.load_state_dict(checkpoint['ema_state_dict'])
            trainer.current_epoch = checkpoint['epoch'] + 1
            trainer.best_val_loss = checkpoint['best_val_loss']
            logger.info(f"Resumed from epoch {trainer.current_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return
    
    # 开始训练
    logger.info("Starting unsupervised training...")
    logger.info("Note: This model does NOT require point-to-point correspondence!")
    logger.info("The model will learn to transfer style while preserving geometry.")
    
    try:
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # 保存中断时的检查点
        trainer.save_checkpoint(is_best=False)
        logger.info("Checkpoint saved")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info(f"Experiment completed. Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()