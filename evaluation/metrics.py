"""
评估指标实现
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Optional
from sklearn.neighbors import NearestNeighbors
import scipy.spatial.distance as scipy_distance


class PointCloudMetrics:
    """点云评估指标"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def chamfer_distance(self, pred: torch.Tensor, target: torch.Tensor,
                        bidirectional: bool = True) -> torch.Tensor:
        """
        计算Chamfer距离
        Args:
            pred: 预测点云 [B, N, 3]
            target: 目标点云 [B, M, 3]
            bidirectional: 是否计算双向距离
        Returns:
            Chamfer距离 [B]
        """
        # 计算距离矩阵
        dist_matrix = torch.cdist(pred, target, p=2)  # [B, N, M]
        
        # pred到target的最近邻距离
        dist_pred_to_target = torch.min(dist_matrix, dim=2)[0]  # [B, N]
        chamfer_pred_to_target = torch.mean(dist_pred_to_target, dim=1)  # [B]
        
        if bidirectional:
            # target到pred的最近邻距离
            dist_target_to_pred = torch.min(dist_matrix, dim=1)[0]  # [B, M]
            chamfer_target_to_pred = torch.mean(dist_target_to_pred, dim=1)  # [B]
            return (chamfer_pred_to_target + chamfer_target_to_pred) / 2
        else:
            return chamfer_pred_to_target
    
    def earth_mover_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Earth Mover's Distance（近似）
        注意：这里使用的是贪心匹配的近似算法
        Args:
            pred: 预测点云 [B, N, 3]
            target: 目标点云 [B, N, 3]（点数必须相同）
        Returns:
            EMD距离 [B]
        """
        assert pred.shape == target.shape, "EMD requires same number of points"
        
        batch_size = pred.shape[0]
        emd_values = []
        
        for b in range(batch_size):
            pred_np = pred[b].cpu().numpy()
            target_np = target[b].cpu().numpy()
            
            # 使用scipy计算成对距离
            dist_matrix = scipy_distance.cdist(pred_np, target_np)
            
            # 贪心匹配
            total_dist = 0
            used_target = set()
            
            for i in range(len(pred_np)):
                # 找到未使用的最近目标点
                min_dist = float('inf')
                best_j = -1
                
                for j in range(len(target_np)):
                    if j not in used_target and dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        best_j = j
                
                if best_j != -1:
                    total_dist += min_dist
                    used_target.add(best_j)
            
            emd_values.append(total_dist / len(pred_np))
        
        return torch.tensor(emd_values, device=self.device, dtype=pred.dtype)
    
    def hausdorff_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Hausdorff距离
        Args:
            pred: 预测点云 [B, N, 3]
            target: 目标点云 [B, M, 3]
        Returns:
            Hausdorff距离 [B]
        """
        dist_matrix = torch.cdist(pred, target, p=2)  # [B, N, M]
        
        # 双向最大最小距离
        max_min_dist_pred = torch.max(torch.min(dist_matrix, dim=2)[0], dim=1)[0]  # [B]
        max_min_dist_target = torch.max(torch.min(dist_matrix, dim=1)[0], dim=1)[0]  # [B]
        
        return torch.max(max_min_dist_pred, max_min_dist_target)
    
    def coverage_score(self, pred: torch.Tensor, target: torch.Tensor,
                      threshold: float = 0.01) -> float:
        """
        计算覆盖度分数
        Args:
            pred: 预测点云 [B, N, 3]
            target: 目标点云 [B, M, 3]
            threshold: 距离阈值
        Returns:
            覆盖度分数
        """
        batch_size = pred.shape[0]
        coverage_scores = []
        
        for b in range(batch_size):
            pred_np = pred[b].cpu().numpy()
            target_np = target[b].cpu().numpy()
            
            # 使用KNN找到每个目标点的最近预测点
            nbrs = NearestNeighbors(n_neighbors=1).fit(pred_np)
            distances, _ = nbrs.kneighbors(target_np)
            
            # 计算被覆盖的目标点比例
            covered = (distances.squeeze() < threshold).sum()
            coverage = covered / len(target_np)
            coverage_scores.append(coverage)
        
        return np.mean(coverage_scores)
    
    def uniformity_score(self, points: torch.Tensor, k: int = 8) -> float:
        """
        计算点云均匀性分数
        Args:
            points: 点云 [B, N, 3]
            k: 近邻数
        Returns:
            均匀性分数（0-1，越高越均匀）
        """
        batch_size = points.shape[0]
        uniformity_scores = []
        
        for b in range(batch_size):
            points_np = points[b].cpu().numpy()
            
            # 计算k近邻距离
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(points_np)
            distances, _ = nbrs.kneighbors(points_np)
            k_distances = distances[:, 1:]  # 排除自身
            
            # 计算距离的标准差
            mean_distances = np.mean(k_distances, axis=1)
            std_distance = np.std(mean_distances)
            mean_distance = np.mean(mean_distances)
            
            # 均匀性分数（变异系数的倒数）
            if mean_distance > 0:
                cv = std_distance / mean_distance
                uniformity = 1.0 / (1.0 + cv)
            else:
                uniformity = 0.0
            
            uniformity_scores.append(uniformity)
        
        return np.mean(uniformity_scores)
    
    def fidelity_score(self, pred: torch.Tensor, target: torch.Tensor,
                      feature_extractor: Optional[nn.Module] = None) -> float:
        """
        计算保真度分数（基于特征相似性）
        Args:
            pred: 预测点云 [B, N, 3]
            target: 目标点云 [B, M, 3]
            feature_extractor: 特征提取器（如PointNet++）
        Returns:
            保真度分数
        """
        if feature_extractor is None:
            # 使用简单的统计特征
            pred_mean = pred.mean(dim=1)  # [B, 3]
            target_mean = target.mean(dim=1)  # [B, 3]
            
            pred_std = pred.std(dim=1)  # [B, 3]
            target_std = target.std(dim=1)  # [B, 3]
            
            # 特征向量
            pred_feat = torch.cat([pred_mean, pred_std], dim=1)  # [B, 6]
            target_feat = torch.cat([target_mean, target_std], dim=1)  # [B, 6]
        else:
            # 使用神经网络特征
            with torch.no_grad():
                pred_feat = feature_extractor(pred)
                target_feat = feature_extractor(target)
        
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(pred_feat, target_feat, dim=1)
        
        return cosine_sim.mean().item()
