import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class ImprovedChunkFusion(nn.Module):
    """改进的块融合模块"""
    
    def __init__(self, overlap_ratio: float = 0.3):
        super().__init__()
        self.overlap_ratio = overlap_ratio
        
        # 边界平滑网络
        self.boundary_smoother = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 3, kernel_size=5, padding=2)
        )
        
        # 权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=1),  # 输入是两个块的拼接
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def find_overlapping_points(self, chunk1: torch.Tensor, chunk2: torch.Tensor, 
                               threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        找到两个块之间的重叠点
        Args:
            chunk1, chunk2: [N1, 3], [N2, 3]
            threshold: 距离阈值
        Returns:
            重叠点的索引
        """
        # 计算点之间的距离
        dist_matrix = torch.cdist(chunk1, chunk2)  # [N1, N2]
        
        # 找到距离小于阈值的点对
        close_pairs = dist_matrix < threshold
        
        # 获取重叠点的索引
        overlap_idx1 = torch.any(close_pairs, dim=1)
        overlap_idx2 = torch.any(close_pairs, dim=0)
        
        return overlap_idx1, overlap_idx2
    
    def smooth_boundary(self, chunk1: torch.Tensor, chunk2: torch.Tensor,
                       overlap_idx1: torch.Tensor, overlap_idx2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        平滑块之间的边界
        """
        # 提取重叠区域
        overlap1 = chunk1[overlap_idx1]  # [M1, 3]
        overlap2 = chunk2[overlap_idx2]  # [M2, 3]
        
        if overlap1.shape[0] == 0 or overlap2.shape[0] == 0:
            return chunk1, chunk2
        
        # 对齐重叠点（使用最近邻匹配）
        dist_matrix = torch.cdist(overlap1, overlap2)
        _, matching_idx = dist_matrix.min(dim=1)
        
        matched_overlap2 = overlap2[matching_idx]  # [M1, 3]
        
        # 计算融合权重
        concat_features = torch.cat([overlap1.T, matched_overlap2.T], dim=0).unsqueeze(0)  # [1, 6, M1]
        weights = self.weight_predictor(concat_features).squeeze()  # [M1]
        
        # 融合重叠点
        fused_overlap = weights.unsqueeze(1) * overlap1 + (1 - weights).unsqueeze(1) * matched_overlap2
        
        # 平滑处理
        fused_overlap_smoothed = self.boundary_smoother(fused_overlap.T.unsqueeze(0)).squeeze(0).T
        
        # 更新原始块
        chunk1_new = chunk1.clone()
        chunk2_new = chunk2.clone()
        
        chunk1_new[overlap_idx1] = fused_overlap_smoothed
        # 对chunk2的重叠部分也进行相应调整
        chunk2_overlap_new = chunk2[overlap_idx2].clone()
        chunk2_overlap_new[matching_idx] = fused_overlap_smoothed
        chunk2_new[overlap_idx2] = chunk2_overlap_new
        
        return chunk1_new, chunk2_new
    
    def merge_all_chunks(self, chunks: List[torch.Tensor], 
                        chunk_positions: List[Tuple[int, int]]) -> torch.Tensor:
        """
        合并所有块成完整点云
        Args:
            chunks: 块列表，每个 [N_chunk, 3]
            chunk_positions: 每个块在原始点云中的位置 (start_idx, end_idx)
        Returns:
            完整点云 [N_total, 3]
        """
        # 首先对相邻块进行边界平滑
        smoothed_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                smoothed_chunks.append(chunks[i])
            else:
                # 找到与前一个块的重叠
                overlap_idx_prev, overlap_idx_curr = self.find_overlapping_points(
                    smoothed_chunks[-1], chunks[i]
                )
                
                # 平滑边界
                chunk_prev_smooth, chunk_curr_smooth = self.smooth_boundary(
                    smoothed_chunks[-1], chunks[i], overlap_idx_prev, overlap_idx_curr
                )
                
                smoothed_chunks[-1] = chunk_prev_smooth
                smoothed_chunks.append(chunk_curr_smooth)
        
        # 计算总点数
        total_points = max(pos[1] for pos in chunk_positions)
        
        # 创建输出张量和权重张量
        merged_points = torch.zeros(total_points, 3, device=chunks[0].device)
        weights = torch.zeros(total_points, device=chunks[0].device)
        
        # 填充点云
        for chunk, (start_idx, end_idx) in zip(smoothed_chunks, chunk_positions):
            num_points = min(len(chunk), end_idx - start_idx)
            
            # 计算该块的权重（中心权重高，边缘权重低）
            chunk_weights = torch.ones(num_points, device=chunk.device)
            fade_size = int(num_points * self.overlap_ratio / 2)
            
            if fade_size > 0:
                # 渐变权重
                fade_in = torch.linspace(0.1, 1.0, fade_size, device=chunk.device)
                fade_out = torch.linspace(1.0, 0.1, fade_size, device=chunk.device)
                
                chunk_weights[:fade_size] = fade_in
                chunk_weights[-fade_size:] = fade_out
            
            # 累加点和权重
            merged_points[start_idx:start_idx + num_points] += chunk[:num_points] * chunk_weights.unsqueeze(1)
            weights[start_idx:start_idx + num_points] += chunk_weights
        
        # 归一化
        weights = weights.clamp(min=1e-8)
        merged_points = merged_points / weights.unsqueeze(1)
        
        return merged_points
