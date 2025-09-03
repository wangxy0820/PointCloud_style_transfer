# models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionLoss(nn.Module):
    """
    Diffusion损失函数：计算预测噪声和真实噪声之间的差异。
    """
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        print(f"Initialized UnifiedDiffusionLoss with {loss_type.upper()} loss.")

    def forward(self, predicted_noise: torch.Tensor, actual_noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted_noise (torch.Tensor): 模型预测的噪声 [B, N, 3]
            actual_noise (torch.Tensor): 实际添加的噪声 [B, N, 3]
        
        Returns:
            torch.Tensor: 一个标量损失值
        """
        return self.loss_fn(predicted_noise, actual_noise)