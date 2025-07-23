import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算Chamfer距离
    Args:
        pred: 预测点云 [B, N, 3]
        target: 目标点云 [B, M, 3]
    Returns:
        两个方向的Chamfer距离
    """
    # 计算距离矩阵
    dist_matrix = torch.cdist(pred, target, p=2)  # [B, N, M]
    
    # 最近邻距离
    dist1 = torch.min(dist_matrix, dim=2)[0]  # [B, N]
    dist2 = torch.min(dist_matrix, dim=1)[0]  # [B, M]
    
    # 平均距离
    chamfer_dist1 = torch.mean(dist1, dim=1)  # [B]
    chamfer_dist2 = torch.mean(dist2, dim=1)  # [B]
    
    return chamfer_dist1, chamfer_dist2


def earth_mover_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算Earth Mover's Distance (近似)
    Args:
        pred: 预测点云 [B, N, 3]
        target: 目标点云 [B, N, 3]
    Returns:
        EMD距离
    """
    batch_size, num_points, _ = pred.shape
    
    # 简化的EMD计算，使用最小二分图匹配的近似
    emd_loss = []
    
    for b in range(batch_size):
        # 计算成本矩阵
        cost_matrix = torch.cdist(pred[b:b+1], target[b:b+1], p=2)[0]  # [N, N]
        
        # 贪心匹配
        total_cost = 0
        used_target = set()
        
        for i in range(num_points):
            min_cost = float('inf')
            best_j = -1
            
            for j in range(num_points):
                if j not in used_target and cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    best_j = j
            
            if best_j != -1:
                total_cost += min_cost
                used_target.add(best_j)
        
        emd_loss.append(total_cost / num_points)
    
    return torch.tensor(emd_loss, device=pred.device, dtype=pred.dtype)


class DiffusionLoss(nn.Module):
    """Diffusion模型的损失函数"""
    
    def __init__(self, 
                 lambda_reconstruction: float = 1.0,
                 lambda_perceptual: float = 0.5,
                 lambda_continuity: float = 0.5,
                 lambda_boundary: float = 1.0):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_perceptual = lambda_perceptual
        self.lambda_continuity = lambda_continuity
        self.lambda_boundary = lambda_boundary
    
    def reconstruction_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """重建损失（MSE）"""
        return F.mse_loss(predicted, target)
    
    def perceptual_loss(self, predicted_features: torch.Tensor, 
                       target_features: torch.Tensor) -> torch.Tensor:
        """感知损失"""
        return F.mse_loss(predicted_features, target_features)
    
    def continuity_loss(self, points: torch.Tensor, k: int = 8) -> torch.Tensor:
        """局部连续性损失"""
        B, N, _ = points.shape
        
        # 计算k近邻
        distances = torch.cdist(points, points)
        _, indices = distances.topk(k=k+1, dim=-1, largest=False)
        indices = indices[:, :, 1:]  # 排除自身
        
        # 获取邻居点
        batch_indices = torch.arange(B).view(B, 1, 1).expand(B, N, k).to(points.device)
        neighbors = points[batch_indices, indices]
        
        # 计算平滑度（邻居点的方差）
        neighbor_mean = neighbors.mean(dim=2, keepdim=True)
        smoothness = ((neighbors - neighbor_mean) ** 2).sum(dim=-1).mean()
        
        return smoothness
    
    def boundary_loss(self, chunk1: torch.Tensor, chunk2: torch.Tensor, 
                     overlap_threshold: float = 0.1) -> torch.Tensor:
        """边界平滑损失"""
        # 找到最近的点对
        dist_matrix = torch.cdist(chunk1, chunk2)
        min_dists, _ = dist_matrix.min(dim=1)
        
        # 只考虑重叠区域的点
        overlap_mask = min_dists < overlap_threshold
        
        if overlap_mask.sum() == 0:
            return torch.tensor(0.0, device=chunk1.device)
        
        # 计算重叠点的距离损失
        boundary_loss = min_dists[overlap_mask].mean()
        
        return boundary_loss
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor,
                predicted_features: Optional[torch.Tensor] = None,
                target_features: Optional[torch.Tensor] = None,
                chunks: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        """
        losses = {}
        
        # 重建损失
        recon_loss = self.reconstruction_loss(predicted, target)
        losses['reconstruction'] = recon_loss
        total_loss = self.lambda_reconstruction * recon_loss
        
        # 感知损失
        if predicted_features is not None and target_features is not None:
            percep_loss = self.perceptual_loss(predicted_features, target_features)
            losses['perceptual'] = percep_loss
            total_loss += self.lambda_perceptual * percep_loss
        
        # 连续性损失
        cont_loss = self.continuity_loss(predicted)
        losses['continuity'] = cont_loss
        total_loss += self.lambda_continuity * cont_loss
        
        # 边界损失
        if chunks is not None and len(chunks) > 1:
            boundary_losses = []
            for i in range(len(chunks) - 1):
                b_loss = self.boundary_loss(chunks[i], chunks[i+1])
                boundary_losses.append(b_loss)
            
            if boundary_losses:
                avg_boundary_loss = torch.stack(boundary_losses).mean()
                losses['boundary'] = avg_boundary_loss
                total_loss += self.lambda_boundary * avg_boundary_loss
        
        losses['total'] = total_loss
        
        return losses


class ChamferLoss(nn.Module):
    """Chamfer距离损失"""
    
    def __init__(self, use_sqrt: bool = False):
        super().__init__()
        self.use_sqrt = use_sqrt
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Chamfer损失
        Args:
            pred: 预测点云 [B, N, 3]
            target: 目标点云 [B, M, 3]
        Returns:
            Chamfer损失
        """
        dist1, dist2 = chamfer_distance(pred, target)
        
        if self.use_sqrt:
            dist1 = torch.sqrt(dist1 + 1e-8)
            dist2 = torch.sqrt(dist2 + 1e-8)
        
        return torch.mean(dist1 + dist2)


class EMDLoss(nn.Module):
    """Earth Mover's Distance损失"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算EMD损失
        Args:
            pred: 预测点云 [B, N, 3]
            target: 目标点云 [B, N, 3]
        Returns:
            EMD损失
        """
        emd = earth_mover_distance(pred, target)
        return torch.mean(emd)


class LocalContinuityLoss(nn.Module):
    """局部连续性损失"""
    
    def __init__(self, k: int = 8, alpha: float = 0.5):
        super().__init__()
        self.k = k
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算局部连续性损失
        Args:
            pred: 预测点云 [B, N, 3]
            target: 目标点云 [B, N, 3]
        Returns:
            局部连续性损失
        """
        batch_size = pred.size(0)
        device = pred.device
        
        total_loss = 0
        
        for b in range(batch_size):
            # 获取单个点云
            pred_pc = pred[b]  # [N, 3]
            target_pc = target[b]  # [N, 3]
            
            # 计算预测点云的k近邻
            pred_dist = torch.cdist(pred_pc.unsqueeze(0), pred_pc.unsqueeze(0))[0]  # [N, N]
            _, pred_idx = torch.topk(pred_dist, k=self.k+1, dim=-1, largest=False)
            pred_idx = pred_idx[:, 1:]  # [N, k], 排除自身
            
            # 计算目标点云的k近邻
            target_dist = torch.cdist(target_pc.unsqueeze(0), target_pc.unsqueeze(0))[0]  # [N, N]
            _, target_idx = torch.topk(target_dist, k=self.k+1, dim=-1, largest=False)
            target_idx = target_idx[:, 1:]  # [N, k], 排除自身
            
            # 计算局部结构差异
            # 1. 局部密度差异
            pred_local_density = torch.mean(pred_dist.gather(1, pred_idx), dim=1)  # [N]
            target_local_density = torch.mean(target_dist.gather(1, target_idx), dim=1)  # [N]
            density_loss = F.mse_loss(pred_local_density, target_local_density)
            
            # 2. 局部方向一致性
            # 计算每个点到其邻居的向量
            pred_neighbors = pred_pc[pred_idx]  # [N, k, 3]
            pred_vectors = pred_neighbors - pred_pc.unsqueeze(1)  # [N, k, 3]
            pred_vectors_norm = F.normalize(pred_vectors, p=2, dim=2)  # 归一化
            
            target_neighbors = target_pc[target_idx]  # [N, k, 3]
            target_vectors = target_neighbors - target_pc.unsqueeze(1)  # [N, k, 3]
            target_vectors_norm = F.normalize(target_vectors, p=2, dim=2)  # 归一化
            
            # 计算局部方向的差异
            direction_similarity = torch.sum(pred_vectors_norm * target_vectors_norm, dim=2)  # [N, k]
            direction_loss = 1.0 - torch.mean(direction_similarity)
            
            # 3. 局部曲率差异（可选）
            # 计算局部协方差矩阵的特征值来估计曲率
            pred_centered = pred_neighbors - pred_pc.unsqueeze(1)  # [N, k, 3]
            pred_cov = torch.bmm(pred_centered.transpose(1, 2), pred_centered) / self.k  # [N, 3, 3]
            
            target_centered = target_neighbors - target_pc.unsqueeze(1)  # [N, k, 3]
            target_cov = torch.bmm(target_centered.transpose(1, 2), target_centered) / self.k  # [N, 3, 3]
            
            # 使用Frobenius范数计算协方差矩阵的差异
            cov_diff = torch.norm(pred_cov - target_cov, p='fro', dim=(1, 2))  # [N]
            curvature_loss = torch.mean(cov_diff)
            
            # 组合损失
            local_loss = density_loss + self.alpha * direction_loss + (1 - self.alpha) * curvature_loss
            total_loss += local_loss
        
        return total_loss / batch_size
