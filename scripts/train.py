#!/usr/bin/env python3
"""
点云风格迁移训练脚本
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.dataset import create_paired_data_loaders
from training.trainer import PointCloudStyleTransferTrainer
from evaluation.metrics import PointCloudMetrics
import logging


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Point Cloud Style Transfer Model')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to processed dataset directory')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # 模型参数
    parser.add_argument('--chunk_size', type=int, default=4096,
                       help='Point cloud chunk size')
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Latent dimension size')
    parser.add_argument('--generator_dim', type=int, default=256,
                       help='Generator style dimension')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate_g', type=float, default=0.0002,
                       help='Generator learning rate')
    parser.add_argument('--learning_rate_d', type=float, default=0.0001,
                       help='Discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Beta2 for Adam optimizer')
    
    # 损失权重
    parser.add_argument('--lambda_recon', type=float, default=10.0,
                       help='Reconstruction loss weight')
    parser.add_argument('--lambda_adv', type=float, default=1.0,
                       help='Adversarial loss weight')
    parser.add_argument('--lambda_cycle', type=float, default=5.0,
                       help='Cycle consistency loss weight')
    parser.add_argument('--lambda_identity', type=float, default=2.0,
                       help='Identity loss weight')
    
    # 设备和输出
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='Directory to save results')
    
    # 训练控制
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Epoch interval to save checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='Epoch interval to run evaluation')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Batch interval to log training progress')
    
    # 数据增强
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--rotation_range', type=float, default=0.1,
                       help='Rotation augmentation range (radians)')
    parser.add_argument('--jitter_std', type=float, default=0.01,
                       help='Jitter augmentation standard deviation')
    parser.add_argument('--scaling_range', type=float, nargs=2, default=[0.9, 1.1],
                       help='Scaling augmentation range')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--experiment_name', type=str, default='',
                       help='Experiment name for logging')
    parser.add_argument('--memory_efficient', action='store_true',
                       help='Enable memory efficient mode')
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_experiment(args):
    """设置实验环境"""
    # 创建实验名称
    if not args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"pointcloud_style_transfer_{timestamp}"
    
    # 创建输出目录
    experiment_dir = os.path.join("experiments", args.experiment_name)
    args.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    args.log_dir = os.path.join(experiment_dir, "logs")
    args.result_dir = os.path.join(experiment_dir, "results")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, 'w') as f:
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")
    
    return experiment_dir


def create_config(args):
    """创建配置对象"""
    config = Config()
    
    # 更新配置 - 确保命令行参数覆盖默认配置
    config.data_root = args.data_dir
    config.processed_data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    
    config.chunk_size = args.chunk_size
    config.latent_dim = args.latent_dim
    config.generator_dim = args.generator_dim
    
    config.num_epochs = args.num_epochs
    config.learning_rate_g = args.learning_rate_g
    config.learning_rate_d = args.learning_rate_d
    config.beta1 = args.beta1
    config.beta2 = args.beta2
    
    config.lambda_recon = args.lambda_recon
    config.lambda_adv = args.lambda_adv
    config.lambda_cycle = args.lambda_cycle
    config.lambda_identity = args.lambda_identity
    
    config.device = f"{args.device}:{args.gpu_id}" if args.device == 'cuda' else args.device
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    config.result_dir = args.result_dir
    
    config.save_interval = args.save_interval
    config.eval_interval = args.eval_interval
    config.log_interval = args.log_interval
    
    # 数据增强参数
    config.use_rotation = args.use_augmentation
    config.use_jitter = args.use_augmentation
    config.use_scaling = args.use_augmentation
    config.rotation_range = args.rotation_range
    config.jitter_std = args.jitter_std
    config.scaling_range = tuple(args.scaling_range)
    
    # 内存优化配置
    if args.memory_efficient:
        print("Memory efficient mode enabled!")
        # 减少模型规模
        config.pointnet_channels = [32, 64, 128, 256]  # 减小通道数
        config.latent_dim = 256  # 减小潜在维度
        config.generator_dim = 128  # 减小生成器维度
        config.discriminator_steps = 3  # 减少判别器训练频率
        config.k = 10  # 减少K近邻数量
        
        # 如果批次大小大于2，自动减小
        if args.batch_size > 2:
            config.batch_size = 2
            print(f"Batch size reduced to 2 for memory efficiency")
    else:
        # 根据批次大小自动调整
        if args.batch_size <= 2:  # 小批次时的内存优化
            config.pointnet_channels = [32, 64, 128, 256]  # 减小通道数
            config.discriminator_steps = 2  # 减少判别器训练频率
        
    print("Applied Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Device: {config.device}")
    print(f"  PointNet channels: {config.pointnet_channels}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Generator dim: {config.generator_dim}")
    
    return config


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置实验环境
    experiment_dir = setup_experiment(args)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Arguments: {vars(args)}")
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name()}")
    
    # 创建配置
    config = create_config(args)
    
    # 准备数据增强参数
    augment_params = None
    if args.use_augmentation:
        augment_params = {
            'rotation_range': args.rotation_range,
            'jitter_std': args.jitter_std,
            'scaling_range': tuple(args.scaling_range)
        }
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_paired_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment_train=args.use_augmentation,
            augment_params=augment_params
        )
        
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # 创建训练器
    logger.info("Creating trainer...")
    try:
        trainer = PointCloudStyleTransferTrainer(config)
        
        # 恢复训练（如果指定）
        if args.resume:
            if os.path.exists(args.resume):
                logger.info(f"Resuming training from {args.resume}")
                trainer.load_checkpoint(args.resume)
            else:
                logger.warning(f"Checkpoint {args.resume} not found, starting from scratch")
        
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return
    
    # 开始训练
    logger.info("Starting training...")
    try:
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint()
        logger.info("Checkpoint saved")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        trainer.save_checkpoint()
        logger.info("Emergency checkpoint saved")
        raise
    
    # 最终评估
    logger.info("Running final evaluation on test set...")
    try:
        # 这里可以添加测试集评估代码
        pass
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
    
    logger.info(f"Experiment completed. Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()