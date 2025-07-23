"""
检查点管理器
"""

import torch
import os
import glob
from typing import Dict, Optional, List


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最多保留的检查点数量
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, state: Dict, epoch: int, 
                       is_best: bool = False,
                       metric: Optional[float] = None,
                       metric_name: str = 'loss'):
        """
        保存检查点
        Args:
            state: 要保存的状态字典
            epoch: 当前epoch
            is_best: 是否是最佳模型
            metric: 指标值
            metric_name: 指标名称
        """
        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(state, latest_path)
        
        # 保存带epoch的检查点
        if metric is not None:
            checkpoint_name = f'checkpoint_epoch_{epoch}_{metric_name}_{metric:.4f}.pth'
        else:
            checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(state, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict:
        """
        加载检查点
        Args:
            checkpoint_path: 检查点路径，如果为None则加载最新的
        Returns:
            状态字典
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return torch.load(checkpoint_path, map_location='cpu')
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新检查点路径"""
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        if os.path.exists(latest_path):
            return latest_path
        
        # 如果没有latest.pth，查找最新的epoch检查点
        checkpoints = self.get_all_checkpoints()
        if checkpoints:
            return checkpoints[-1]  # 返回最新的
        
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点路径"""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_path):
            return best_path
        return None
    
    def get_all_checkpoints(self) -> List[str]:
        """获取所有检查点路径"""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
        checkpoints = sorted(glob.glob(pattern))
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最新的几个"""
        checkpoints = self.get_all_checkpoints()
        
        # 保留latest、best和最新的max_checkpoints个
        preserved = [
            os.path.join(self.checkpoint_dir, 'latest.pth'),
            os.path.join(self.checkpoint_dir, 'best_model.pth')
        ]
        
        # 需要删除的检查点
        to_delete = []
        for checkpoint in checkpoints:
            if checkpoint not in preserved:
                to_delete.append(checkpoint)
        
        # 只保留最新的max_checkpoints个
        if len(to_delete) > self.max_checkpoints:
            for checkpoint in to_delete[:-self.max_checkpoints]:
                os.remove(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")
