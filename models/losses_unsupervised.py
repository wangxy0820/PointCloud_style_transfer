# training/losses_unsupervised.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    计算Chamfer距离.
    Args:
        p1: 点云1 [B, N, 3]
        p2: 点云2 [B, M, 3]
    Returns:
        Chamfer距离 [B]
    """
    dist_matrix = torch.cdist(p1, p2, p=1.0)
    dist1 = torch.min(dist_matrix, dim=2)[0]  # p1 -> p2
    dist2 = torch.min(dist_matrix, dim=1)[0]  # p2 -> p1
    return (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2.0

class UnsupervisedDiffusionLoss(nn.Module):
    """
    为无监督点云风格迁移设计的、基于Chamfer Loss的、更稳健的Diffusion损失.
    """
    def __init__(self, 
                 lambda_diffusion: float = 1.0,
                 lambda_chamfer: float = 5.0,
                 lambda_content: float = 2.0,
                 lambda_style: float = 0.01,
                 lambda_smooth: float = 0.1,
                 lambda_lidar_structure: float = 0.5):
        super().__init__()
        self.lambda_diffusion = lambda_diffusion
        self.lambda_chamfer = lambda_chamfer
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_smooth = lambda_smooth
        self.lambda_lidar_structure = lambda_lidar_structure

    def content_preservation_loss(self, content_original: torch.Tensor, 
                                 content_from_noisy: torch.Tensor) -> torch.Tensor:
        """确保内容编码器提取的特征对噪声具有鲁棒性"""
        return F.mse_loss(content_from_noisy, content_original)

    def lidar_structure_loss(self, generated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """LiDAR特定的结构保持损失 (保持不变，因为其设计合理)"""
        gen_z_mean, gen_z_std = generated[:, :, 2].mean(), generated[:, :, 2].std()
        orig_z_mean, orig_z_std = original[:, :, 2].mean(), original[:, :, 2].std()
        height_loss = F.l1_loss(gen_z_mean, orig_z_mean) + F.l1_loss(gen_z_std, orig_z_std)
        return height_loss

    def style_consistency_loss(self, style_sim: torch.Tensor, style_real: torch.Tensor) -> torch.Tensor:
        """风格一致性损失 (保持不变)"""
        return F.mse_loss(style_sim, style_real)

    def smoothness_loss(self, generated: torch.Tensor) -> torch.Tensor:
        """平滑性损失，惩罚局部剧烈变化 (保持不变)"""
        diff = generated[:, 1:] - generated[:, :-1]
        diff_norm = torch.norm(diff, p=2, dim=2)
        return diff_norm.std(dim=1).mean()

    def forward(self, 
                pred_noise: torch.Tensor,
                target_noise: torch.Tensor,
                generated_points: torch.Tensor, # 这是去噪后的 x0_pred
                original_points: torch.Tensor,  # 这是干净的 x_start
                content_original: torch.Tensor,
                content_from_noisy: torch.Tensor,
                style_source: torch.Tensor,
                style_target: torch.Tensor,
                warmup_factor: float = 1.0) -> Dict[str, torch.Tensor]:
        
        losses = {}

        # 1. 基础Diffusion损失 (L2)
        losses['diffusion'] = F.mse_loss(pred_noise, target_noise)

        # 2. 几何/结构保持损失 (核心修改)
        #    使用Chamfer Distance确保生成的点云在宏观和微观上都与原始点云相似
        losses['chamfer'] = chamfer_distance(generated_points, original_points).mean()

        # 3. 内容保持损失
        losses['content'] = self.content_preservation_loss(content_original, content_from_noisy)
        
        # 4. LiDAR结构保持损失
        losses['lidar_structure'] = self.lidar_structure_loss(generated_points, original_points)

        # 5. 风格迁移损失
        #    注意：这里的style_source是生成点云的风格，style_target是参考域的风格
        #    为了让模型学习转换，应该让生成点云的风格接近目标风格
        generated_style = style_source # 这里的style_source实际是从generated_points提取的
        losses['style'] = self.style_consistency_loss(generated_style, style_target)
        
        # 6. 平滑度损失
        losses['smooth'] = self.smoothness_loss(generated_points)

        # 计算总损失
        # 其他损失项使用warmup_factor，让模型在早期专注于学习基础的去噪任务
        total_loss = (
            self.lambda_diffusion * losses['diffusion'] +
            warmup_factor * (
                self.lambda_chamfer * losses['chamfer'] +
                self.lambda_content * losses['content'] +
                self.lambda_lidar_structure * losses['lidar_structure'] +
                self.lambda_style * losses['style'] +
                self.lambda_smooth * losses['smooth']
            )
        )
        
        losses['total'] = total_loss
        
        return losses