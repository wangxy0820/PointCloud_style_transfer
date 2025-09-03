# scripts/train.py

import argparse
import os
import sys
import torch
import random
import numpy as np
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.dataset import create_dataloaders
from training.trainer import DiffusionTrainer
from utils.logger import Logger

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Unified Conditional Point Cloud Diffusion Training')
    
    # 简化命令行参数，主要通过config文件控制
    parser.add_argument('--experiment_name', type=str, default=None, help='Override experiment name in config.')
    parser.add_argument('--batch_size', type=int, default=8, help='Override batch size in config.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # 加载配置并允许命令行覆盖
    config = Config()
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.batch_size:
        config._batch_size = args.batch_size
        
    logger = Logger('train_main', log_dir=config.log_dir, experiment_name=config.experiment_name)
    logger.info(f"Hierarchical mode enabled: {config.use_hierarchical}")
    
    # 创建数据加载器
    try:
        train_loader, val_loader = create_dataloaders(
            config.processed_data_dir, config.batch_size, config.num_workers
        )
        logger.info(f"Data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}\n{traceback.format_exc()}")
        return
        
    # 创建并启动训练器
    try:
        trainer = DiffusionTrainer(config=config, device=args.device)
        logger.info(f"Trainer created for experiment: {config.experiment_name}")
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}\n{traceback.format_exc()}")

if __name__ == '__main__':
    main()