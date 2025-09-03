# utils/checkpoint.py

import torch
import os
import glob
from typing import Optional

# 确保可以正确导入
from utils.logger import Logger
from utils.ema import ExponentialMovingAverage

class CheckpointManager:
    """
    Checkpoint管理器。
    """
    def __init__(self, checkpoint_dir: str, experiment_name: str):
        """
        构造函数。
        
        Args:
            checkpoint_dir (str): 保存checkpoint的根目录 (e.g., 'checkpoints')
            experiment_name (str): 当前实验的名称
        """
        self.base_dir = os.path.join(checkpoint_dir, experiment_name)
        os.makedirs(self.base_dir, exist_ok=True)
        self.logger = Logger(name='CheckpointManager', log_dir='logs', experiment_name=experiment_name)

    def save(self, model, optimizer: torch.optim.Optimizer, 
             ema: Optional[ExponentialMovingAverage], epoch: int, is_best: bool = False):
        """
        保存模型的状态。
        修复：确保config正确保存
        """
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model.config if hasattr(model, 'config') else None, # 保存配置以备推理时使用
        }
        
        # 保存EMA状态
        if ema:
            try:
                state['ema_state_dict'] = ema.state_dict()
                self.logger.info("EMA state saved in checkpoint")
            except Exception as e:
                self.logger.warning(f"Failed to save EMA state: {e}")

        # 保存常规的checkpoint
        filename = f"ckpt_epoch_{epoch:04d}.pth"
        filepath = os.path.join(self.base_dir, filename)
        
        try:
            torch.save(state, filepath)
            self.logger.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise

        # 如果是最佳模型，额外保存一份名为best_model.pth
        if is_best:
            best_filepath = os.path.join(self.base_dir, "best_model.pth")
            try:
                torch.save(state, best_filepath)
                self.logger.info(f"Best model updated and saved to {best_filepath}")
            except Exception as e:
                self.logger.error(f"Failed to save best model: {e}")

    def load(self, model, optimizer: torch.optim.Optimizer, 
             ema: Optional[ExponentialMovingAverage]) -> int:
        """
        加载最新的checkpoint，并返回当前epoch。
        """
        latest_ckpt = self._find_latest_checkpoint()
        if not latest_ckpt:
            self.logger.info("No checkpoint found. Starting training from scratch.")
            return 0
            
        self.logger.info(f"Loading checkpoint from {latest_ckpt}")
        
        try:
            # 允许加载在不同设备上保存的模型
            checkpoint = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
            
            # 验证checkpoint内容
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                self.logger.error(f"Checkpoint missing required keys: {missing_keys}")
                return 0
            
            # 加载模型权重
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("Model state loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model state: {e}")
                return 0
            
            # 加载优化器状态
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # 将优化器状态移动到正确设备
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(model.parameters().__next__().device)
                self.logger.info("Optimizer state loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer state: {e}, continuing with fresh optimizer")
            
            # 加载EMA状态
            if ema and 'ema_state_dict' in checkpoint:
                try:
                    ema.load_state_dict(checkpoint['ema_state_dict'])
                    self.logger.info("EMA state loaded successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to load EMA state: {e}, continuing with fresh EMA")
            
            epoch_start = checkpoint.get('epoch', -1) + 1
            self.logger.info(f"Resuming training from epoch {epoch_start}")
            return epoch_start
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            return 0

    def _find_latest_checkpoint(self) -> Optional[str]:
        """寻找目录下最新的checkpoint文件"""
        ckpt_files = glob.glob(os.path.join(self.base_dir, "ckpt_epoch_*.pth"))
        if not ckpt_files:
            return None
        
        # 根据文件名中的epoch数字排序，选择最新的
        def extract_epoch(filepath):
            try:
                filename = os.path.basename(filepath)
                epoch_str = filename.replace('ckpt_epoch_', '').replace('.pth', '')
                return int(epoch_str)
            except:
                return -1
                
        latest_file = max(ckpt_files, key=extract_epoch)
        return latest_file
    
    def get_best_model_path(self) -> Optional[str]:
        """获取最佳模型的路径"""
        best_path = os.path.join(self.base_dir, "best_model.pth")
        if os.path.exists(best_path):
            return best_path
        return None