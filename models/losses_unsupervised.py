import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """计算Chamfer距离 (L1版本，对离群点更鲁棒)"""
    dist_matrix = torch.cdist(p1, p2, p=1.0)
    dist1 = torch.min(dist_matrix, dim=2)[0]
    dist2 = torch.min(dist_matrix, dim=1)[0]
    return (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2.0

class UnsupervisedDiffusionLoss(nn.Module):
    """
    实现了课程学习的、更稳健的Diffusion损失函数。
    """
    def __init__(self, 
                 lambda_diffusion: float = 1.0,
                 lambda_chamfer: float = 10.0,
                 lambda_content: float = 1.0,
                 lambda_style: float = 0.0,
                 lambda_smooth: float = 0.5,
                 lambda_lidar_structure: float = 1.0):
        super().__init__()
        self.lambda_diffusion = lambda_diffusion
        self.lambda_chamfer = lambda_chamfer # 这是最终的目标权重
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_smooth = lambda_smooth
        self.lambda_lidar_structure = lambda_lidar_structure

    def content_preservation_loss(self, c_orig: torch.Tensor, c_noisy: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(c_noisy, c_orig)

    def lidar_structure_loss(self, gen: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
        gen_z_mean, gen_z_std = gen[:, :, 2].mean(), gen[:, :, 2].std()
        orig_z_mean, orig_z_std = orig[:, :, 2].mean(), orig[:, :, 2].std()
        return F.l1_loss(gen_z_mean, orig_z_mean) + F.l1_loss(gen_z_std, orig_z_std)

    def style_consistency_loss(self, style_gen: torch.Tensor, style_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(style_gen, style_target)

    def smoothness_loss(self, gen: torch.Tensor) -> torch.Tensor:
        diff = gen[:, 1:] - gen[:, :-1]
        return torch.norm(diff, p=2, dim=2).std(dim=1).mean()

    def forward(self, 
                pred_noise: torch.Tensor,
                target_noise: torch.Tensor,
                generated_points: torch.Tensor,
                original_points: torch.Tensor,
                content_original: torch.Tensor,
                content_from_noisy: torch.Tensor,
                style_source: torch.Tensor,
                style_target: torch.Tensor,
                warmup_factor: float = 1.0,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        
        losses = {}

        # 1. 基础Diffusion损失 (永远存在)
        losses['diffusion'] = F.mse_loss(pred_noise, target_noise)

        # 2. 几何/结构保持损失 (使用课程学习动态调整权重)
        chamfer_warmup_epochs = 10  # 前10个epoch完全不使用chamfer loss
        ramp_up_duration = 20       # 在接下来的20个epoch中，权重从0线性增加到目标值
        
        if epoch < chamfer_warmup_epochs:
            effective_chamfer_lambda = 0.0
        else:
            progress = min(1.0, (epoch - chamfer_warmup_epochs) / ramp_up_duration)
            effective_chamfer_lambda = self.lambda_chamfer * progress
        
        losses['chamfer'] = chamfer_distance(generated_points, original_points).mean()
        
        # 其他辅助损失
        losses['content'] = self.content_preservation_loss(content_original, content_from_noisy)
        losses['lidar_structure'] = self.lidar_structure_loss(generated_points, original_points)
        losses['style'] = self.style_consistency_loss(style_source, style_target)
        losses['smooth'] = self.smoothness_loss(generated_points)

        # 计算总损失
        total_loss = (
            self.lambda_diffusion * losses['diffusion'] +
            effective_chamfer_lambda * losses['chamfer'] + # 使用动态权重
            warmup_factor * ( # 其他辅助损失仍然使用简单的warmup
                self.lambda_content * losses['content'] +
                self.lambda_lidar_structure * losses['lidar_structure'] +
                self.lambda_style * losses['style'] +
                self.lambda_smooth * losses['smooth']
            )
        )
        
        losses['total'] = total_loss
        losses['debug_chamfer_lambda'] = torch.tensor(effective_chamfer_lambda, device=pred_noise.device) # 用于日志记录
        
        return losses
