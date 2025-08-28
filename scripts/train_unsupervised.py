import argparse
import os
import sys
import torch
import numpy as np
import random
import traceback # ADDED: 导入traceback模块用于打印详细错误

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_unsupervised import ConfigUnsupervised
from data.dataset import create_dataloaders
from training.trainer_unsupervised import UnsupervisedDiffusionTrainer
from utils.logger import Logger

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Two-Stage Unsupervised Point Cloud Style Transfer')
    
    parser.add_argument('--experiment_name', type=str, default=None, help='Name for the experiment directory.')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='Training stage: 1 for reconstruction, 2 for style transfer.')
    parser.add_argument('--stage1_checkpoint', type=str, default=None, help='Path to the best model from stage 1, required for stage 2.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use.')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = ConfigUnsupervised()
    #config.experiment_name = args.experiment_name
    config.device = args.device
    
    if args.stage == 2:
        print("--- CONFIGURING FOR STAGE 2: STYLE TRANSFER FINETUNING ---")
        config.learning_rate = 1e-5
        config.lambda_chamfer = 5.0
        config.lambda_style = 0.05
        config.lambda_content = 0.0
        if not args.stage1_checkpoint:
            raise ValueError("--stage1_checkpoint is required for stage 2 training.")
    else:
        print("--- CONFIGURING FOR STAGE 1: GEOMETRIC RECONSTRUCTION ---")

    experiment_dir = os.path.join('experiments', config.experiment_name)
    log_dir = os.path.join(experiment_dir, 'logs')
    config.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    config.log_dir = log_dir
    config.result_dir = os.path.join(experiment_dir, 'results')
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    logger = Logger(name='train_script', log_dir=log_dir, log_level='INFO')
    
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Selected Stage: {args.stage}")
    logger.info(f"Full Configuration: {config}")
    
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader, _ = create_dataloaders(
            config.processed_data_dir,
            config.batch_size,
            config.num_workers,
            config.chunk_size,
            config=config
        )
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        # CHANGED: 修改了错误记录方式以兼容您的Logger
        logger.error(f"Failed to create data loaders: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
        
    logger.info("Creating unsupervised trainer...")
    try:
        trainer = UnsupervisedDiffusionTrainer(
            config=config,
            stage=args.stage,
            stage1_checkpoint_path=args.stage1_checkpoint
        )
    except Exception as e:
        # CHANGED: 修改了错误记录方式
        logger.error(f"Failed to create trainer: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
        
    logger.info(f"Starting Stage {args.stage} training...")
    try:
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving checkpoint...")
        trainer.save_checkpoint(is_best=False)
        logger.info("Checkpoint saved.")
    except Exception as e:
        # CHANGED: 修改了错误记录方式
        logger.error(f"Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
