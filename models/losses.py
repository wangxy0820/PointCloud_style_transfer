# models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

def chamfer_distance_chunked_optimized(pred: torch.Tensor, target: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """
    分块计算Chamfer距离的最终优化版，使用矩阵乘法避免巨大中间张量。
    这是解决OOM问题的关键。
    Args:
        pred: [B, N, 3] 预测点云
        target: [B, M, 3] 目标点云
        chunk_size: int, 每次处理多少个点
    Returns:
        chamfer_dist: [B] 每个batch的Chamfer距离
    """
    B, N, _ = pred.shape
    _, M, _ = target.shape
    device = pred.device

    # 预计算每个点的平方范数
    pred_sq = torch.sum(pred ** 2, dim=-1, keepdim=True)  # [B, N, 1]
    target_sq = torch.sum(target ** 2, dim=-1, keepdim=True).transpose(1, 2) # [B, 1, M]

    # pred到target的最短距离
    dist_pred_to_target = torch.zeros(B, N, device=device)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        pred_chunk = pred[:, i:end, :]  # [B, chunk_size, 3]
        pred_sq_chunk = pred_sq[:, i:end, :] # [B, chunk_size, 1]
        
        # 使用 a^2 - 2ab + b^2 公式
        # -2ab term
        matmul_term = -2 * torch.bmm(pred_chunk, target.transpose(1, 2)) # [B, chunk_size, M]
        # a^2 + b^2 term
        dist_matrix_chunk = pred_sq_chunk + target_sq + matmul_term
        dist_matrix_chunk = torch.clamp(dist_matrix_chunk, min=0) # 避免数值误差导致负数
        
        min_dist_chunk = torch.min(dist_matrix_chunk, dim=2)[0]
        dist_pred_to_target[:, i:end] = min_dist_chunk

    # target到pred的最短距离
    dist_target_to_pred = torch.zeros(B, M, device=device)
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        target_chunk = target[:, i:end, :] # [B, chunk_size, 3]
        target_sq_chunk = target_sq[:, :, i:end].transpose(1, 2) # [B, chunk_size, 1]

        # 使用 a^2 - 2ab + b^2 公式
        # -2ab term
        matmul_term = -2 * torch.bmm(target_chunk, pred.transpose(1, 2)) # [B, chunk_size, N]
        # a^2 + b^2 term
        dist_matrix_chunk = target_sq_chunk + pred_sq.transpose(1, 2) + matmul_term
        dist_matrix_chunk = torch.clamp(dist_matrix_chunk, min=0)
        
        min_dist_chunk = torch.min(dist_matrix_chunk, dim=2)[0]
        dist_target_to_pred[:, i:end] = min_dist_chunk
    
    chamfer_dist = torch.mean(dist_pred_to_target, dim=1) + torch.mean(dist_target_to_pred, dim=1)
    
    return chamfer_dist


class DiffusionLoss(nn.Module):
    """
    扩散模型损失函数
    """
    def __init__(self, 
                 noise_weight: float = 1.0,
                 chamfer_weight: float = 0.1):
        super().__init__()
        self.noise_weight = noise_weight
        self.chamfer_weight = chamfer_weight
        
        print(f"DiffusionLoss initialized:")
        print(f"  Noise L1 weight: {noise_weight}")
        print(f"  Chamfer weight: {chamfer_weight}")
        
    def forward(self, predicted_noise: torch.Tensor, actual_noise: torch.Tensor,
                predicted_points_coarse: torch.Tensor = None, 
                target_points_coarse: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        """
        loss_dict = {}
        
        # 主要损失：噪声预测损失（L1损失）
        noise_loss = F.l1_loss(predicted_noise, actual_noise)
        total_loss = self.noise_weight * noise_loss
        
        loss_dict['noise_loss'] = noise_loss.item()
        
        # 如果提供了下采样点云，计算Chamfer距离损失
        if self.chamfer_weight > 0 and predicted_points_coarse is not None and target_points_coarse is not None:
            # 调用最终显存优化版的chamfer distance
            chamfer_loss = torch.mean(chamfer_distance_chunked_optimized(predicted_points_coarse, target_points_coarse))
            total_loss += self.chamfer_weight * chamfer_loss
            loss_dict['chamfer_loss'] = chamfer_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict