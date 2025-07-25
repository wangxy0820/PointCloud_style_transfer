import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class GeometryPreservingDiffusionLoss(nn.Module):
    """专门为保持几何形状设计的Diffusion损失函数"""
    
    def __init__(self, 
                 lambda_diffusion: float = 1.0,
                 lambda_shape: float = 2.0,
                 lambda_local: float = 1.0,
                 lambda_content: float = 1.0,    # 添加内容损失权重
                 lambda_style: float = 0.1,
                 lambda_smooth: float = 0.5):
        super().__init__()
        self.lambda_diffusion = lambda_diffusion
        self.lambda_shape = lambda_shape
        self.lambda_local = lambda_local
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_smooth = lambda_smooth
    
    def diffusion_loss(self, pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        """基础Diffusion损失 - 学习去噪"""
        return F.mse_loss(pred_noise, target_noise)
    
    def shape_preservation_loss(self, generated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """形状保持损失 - 确保生成的点云保持原始形状"""
        batch_size = generated.shape[0]
        
        # 1. 全局形状保持 - 点云的整体分布
        # 计算点云中心
        gen_center = generated.mean(dim=1, keepdim=True)  # [B, 1, 3]
        orig_center = original.mean(dim=1, keepdim=True)
        center_loss = F.mse_loss(gen_center, orig_center)
        
        # 2. 点云的空间范围保持
        # 计算各个维度的范围
        gen_min = generated.min(dim=1)[0]  # [B, 3]
        gen_max = generated.max(dim=1)[0]
        orig_min = original.min(dim=1)[0]
        orig_max = original.max(dim=1)[0]
        
        range_loss = F.mse_loss(gen_max - gen_min, orig_max - orig_min)
        
        # 3. 点云的主方向保持（使用简化的PCA）
        # 中心化
        gen_centered = generated - gen_center
        orig_centered = original - orig_center
        
        # 计算协方差矩阵
        gen_cov = torch.bmm(gen_centered.transpose(1, 2), gen_centered) / (generated.shape[1] - 1)
        orig_cov = torch.bmm(orig_centered.transpose(1, 2), orig_centered) / (original.shape[1] - 1)
        
        # 使用Frobenius范数比较协方差矩阵
        cov_loss = torch.norm(gen_cov - orig_cov, p='fro', dim=(1, 2)).mean()
        
        # 4. 点云密度分布保持
        # 计算每个点到中心的距离分布
        gen_dists = torch.norm(gen_centered, dim=2)  # [B, N]
        orig_dists = torch.norm(orig_centered, dim=2)
        
        # 比较距离分布的统计量
        gen_dist_mean = gen_dists.mean(dim=1)
        gen_dist_std = gen_dists.std(dim=1)
        orig_dist_mean = orig_dists.mean(dim=1)
        orig_dist_std = orig_dists.std(dim=1)
        
        dist_loss = F.mse_loss(gen_dist_mean, orig_dist_mean) + F.mse_loss(gen_dist_std, orig_dist_std)
        
        # 组合所有形状损失
        total_shape_loss = center_loss + range_loss * 0.5 + cov_loss * 0.1 + dist_loss * 0.5
        
        return total_shape_loss
    
    def local_structure_loss(self, generated: torch.Tensor, original: torch.Tensor, k: int = 16) -> torch.Tensor:
        """局部结构损失 - 保持局部几何关系"""
        batch_size = generated.shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            # 计算原始点云的k近邻
            orig_dists = torch.cdist(original[b], original[b])  # [N, N]
            _, orig_indices = torch.topk(orig_dists, k=k+1, dim=1, largest=False)
            orig_indices = orig_indices[:, 1:]  # 排除自身
            
            # 对于每个点，计算其k近邻的相对位置
            orig_neighbors = original[b][orig_indices]  # [N, k, 3]
            orig_relative = orig_neighbors - original[b].unsqueeze(1)  # [N, k, 3]
            
            # 生成点云的对应关系（假设点的顺序保持）
            gen_neighbors = generated[b][orig_indices]  # [N, k, 3]
            gen_relative = gen_neighbors - generated[b].unsqueeze(1)  # [N, k, 3]
            
            # 比较相对位置
            local_loss = F.mse_loss(gen_relative, orig_relative)
            total_loss += local_loss
        
        return total_loss / batch_size
    
    def style_consistency_loss(self, style_sim: torch.Tensor, style_real: torch.Tensor) -> torch.Tensor:
        """风格一致性损失 - 学习目标域的风格"""
        # 使用余弦相似度，让不同域的风格有所区别
        cosine_sim = F.cosine_similarity(style_sim, style_real, dim=1)
        # 我们希望风格有区别但不要完全相反
        return (cosine_sim - 0.5).abs().mean()
    
    def smoothness_loss(self, generated: torch.Tensor) -> torch.Tensor:
        """平滑性损失 - 避免生成的点云有异常的尖锐变化"""
        # 计算点云的局部平滑度
        batch_size = generated.shape[0]
        smooth_loss = 0.0
        
        for b in range(batch_size):
            # 计算最近的几个邻居
            dists = torch.cdist(generated[b], generated[b])
            _, indices = torch.topk(dists, k=5, dim=1, largest=False)
            
            # 计算到邻居的平均距离
            neighbor_dists = dists.gather(1, indices[:, 1:])  # 排除自身
            mean_dists = neighbor_dists.mean(dim=1)
            
            # 平滑度：邻居距离的方差应该小
            smooth_loss += mean_dists.std()
        
        return smooth_loss / batch_size
    
    def content_preservation_loss(self, content_original: torch.Tensor, 
                                 content_from_noisy: torch.Tensor) -> torch.Tensor:
        """内容保持损失 - 确保内容编码器提取的特征一致"""
        # 1. 基础MSE损失
        mse_loss = F.mse_loss(content_from_noisy, content_original)
        
        # 2. 确保内容特征有空间变化（不是常数）
        # 计算空间方差
        orig_spatial_var = content_original.var(dim=2, keepdim=True).mean()
        noisy_spatial_var = content_from_noisy.var(dim=2, keepdim=True).mean()
        
        # 如果方差太小，说明内容编码器输出了常数
        var_loss = F.relu(0.1 - orig_spatial_var) + F.relu(0.1 - noisy_spatial_var)
        
        # 3. 特征激活损失 - 防止内容特征死亡
        activation_loss = F.relu(1.0 - content_original.abs().mean()) * 0.1
        
        return mse_loss + var_loss + activation_loss
    
    def forward(self, 
                pred_noise: torch.Tensor,
                target_noise: torch.Tensor,
                generated_points: Optional[torch.Tensor] = None,
                original_points: Optional[torch.Tensor] = None,
                content_original: Optional[torch.Tensor] = None,
                content_from_noisy: Optional[torch.Tensor] = None,
                style_source: Optional[torch.Tensor] = None,
                style_target: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失（添加内容损失）
        """
        losses = {}
        
        # 1. 基础Diffusion损失
        diff_loss = self.diffusion_loss(pred_noise, target_noise)
        losses['diffusion'] = diff_loss
        total_loss = self.lambda_diffusion * diff_loss
        
        # 2. 内容保持损失（新增）
        if content_original is not None and content_from_noisy is not None:
            content_loss = self.content_preservation_loss(content_original, content_from_noisy)
            losses['content'] = content_loss
            total_loss += self.lambda_content * content_loss
        
        # 3. 形状保持损失
        if generated_points is not None and original_points is not None:
            shape_loss = self.shape_preservation_loss(generated_points, original_points)
            losses['shape'] = shape_loss
            total_loss += self.lambda_shape * shape_loss
            
            # 4. 局部结构损失
            local_loss = self.local_structure_loss(generated_points, original_points)
            losses['local'] = local_loss
            total_loss += self.lambda_local * local_loss
            
            # 5. 平滑性损失
            smooth_loss = self.smoothness_loss(generated_points)
            losses['smooth'] = smooth_loss
            total_loss += self.lambda_smooth * smooth_loss
        
        # 6. 风格损失
        if style_source is not None and style_target is not None:
            style_loss = self.style_consistency_loss(style_source, style_target)
            losses['style'] = style_loss
            total_loss += self.lambda_style * style_loss
        
        losses['total'] = total_loss
        
        return losses