import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class GeometryPreservingDiffusionLoss(nn.Module):
    """专门为保持几何形状设计的Diffusion损失函数 - LiDAR增强版"""
    
    def __init__(self, 
                 lambda_diffusion: float = 1.0,
                 lambda_shape: float = 2.0,
                 lambda_local: float = 1.0,
                 lambda_content: float = 2.0,    # 增加内容损失权重
                 lambda_style: float = 0.02,     # 减小风格权重
                 lambda_smooth: float = 0.5,
                 lambda_lidar_structure: float = 0.5):  # 新增LiDAR结构损失
        super().__init__()
        self.lambda_diffusion = lambda_diffusion
        self.lambda_shape = lambda_shape
        self.lambda_local = lambda_local
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_smooth = lambda_smooth
        self.lambda_lidar_structure = lambda_lidar_structure
        
        # 添加损失统计（用于自适应权重）
        self.register_buffer('loss_ema', torch.zeros(7))  # 7个损失项
        self.register_buffer('ema_count', torch.tensor(0))
        self.ema_decay = 0.99
    
    def update_loss_ema(self, losses: Dict[str, torch.Tensor]):
        """更新损失的指数移动平均"""
        loss_values = torch.stack([
            losses.get('diffusion', torch.tensor(0.0)),
            losses.get('content', torch.tensor(0.0)),
            losses.get('shape', torch.tensor(0.0)),
            losses.get('local', torch.tensor(0.0)),
            losses.get('smooth', torch.tensor(0.0)),
            losses.get('lidar_structure', torch.tensor(0.0)),
            losses.get('style', torch.tensor(0.0))
        ]).detach()
        
        if self.ema_count == 0:
            self.loss_ema = loss_values
        else:
            self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * loss_values
        
        self.ema_count += 1
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """根据损失的相对大小自适应调整权重"""
        if self.ema_count < 10:
            # 初始阶段使用原始权重
            return {
                'diffusion': self.lambda_diffusion,
                'content': self.lambda_content,
                'shape': self.lambda_shape,
                'local': self.lambda_local,
                'smooth': self.lambda_smooth,
                'lidar_structure': self.lambda_lidar_structure,
                'style': self.lambda_style
            }
        
        # 计算相对比例
        base_loss = self.loss_ema[0] + 1e-8  # diffusion loss作为基准
        relative_scale = self.loss_ema / base_loss
        
        # 原始权重
        weights = {
            'diffusion': self.lambda_diffusion,
            'content': self.lambda_content,
            'shape': self.lambda_shape,
            'local': self.lambda_local,
            'smooth': self.lambda_smooth,
            'lidar_structure': self.lambda_lidar_structure,
            'style': self.lambda_style
        }
        
        # 如果LiDAR损失相对过大（超过diffusion损失的10倍），降低其权重
        if relative_scale[5] > 10:
            weights['lidar_structure'] *= 0.1
            if self.ema_count % 100 == 0:  # 每100次打印一次警告
                print(f"Warning: LiDAR loss too large ({relative_scale[5]:.1f}x diffusion loss), reducing weight to {weights['lidar_structure']:.3f}")
        
        return weights
    
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
    
    def lidar_structure_loss(self, generated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """LiDAR特定的结构保持损失 - 简化和稳定版本"""
        batch_size = generated.shape[0]
        device = generated.device
        
        # 使用向量化操作，避免循环
        gen = generated.view(-1, 3)  # [B*N, 3]
        orig = original.view(-1, 3)  # [B*N, 3]
        
        # 1. 简单的高度分布保持（Z轴）
        gen_z = gen[:, 2]
        orig_z = orig[:, 2]
        
        # 使用L1损失，更稳定
        z_mean_loss = F.l1_loss(gen_z.mean(), orig_z.mean())
        z_std_loss = F.l1_loss(gen_z.std(), orig_z.std())
        height_loss = (z_mean_loss + z_std_loss) * 0.5
        
        # 2. XY平面分布保持
        gen_xy = gen[:, :2]
        orig_xy = orig[:, :2]
        
        # XY范围
        gen_xy_min = gen_xy.min(dim=0)[0]
        gen_xy_max = gen_xy.max(dim=0)[0]
        orig_xy_min = orig_xy.min(dim=0)[0]
        orig_xy_max = orig_xy.max(dim=0)[0]
        
        xy_range_loss = F.l1_loss(gen_xy_max - gen_xy_min, orig_xy_max - orig_xy_min)
        
        # 3. 局部平滑性（简化版）
        # 只计算前面一部分点，避免内存爆炸
        n_points = generated.shape[1]
        n_check = min(100, n_points - 1)
        
        gen_batch = generated.view(batch_size, n_points, 3)
        orig_batch = original.view(batch_size, n_points, 3)
        
        # 相邻点差异
        gen_diff = gen_batch[:, 1:n_check+1] - gen_batch[:, :n_check]
        orig_diff = orig_batch[:, 1:n_check+1] - orig_batch[:, :n_check]
        
        # 相邻点距离
        gen_dist = torch.norm(gen_diff, dim=2).mean()
        orig_dist = torch.norm(orig_diff, dim=2).mean()
        smoothness_loss = F.l1_loss(gen_dist, orig_dist)
        
        # 组合LiDAR损失 - 使用更小的权重
        total_lidar_loss = (
            height_loss * 0.4 +
            xy_range_loss * 0.3 +
            smoothness_loss * 0.3
        )
        
        # 限制最大值，防止爆炸
        total_lidar_loss = torch.clamp(total_lidar_loss, max=5.0)
        
        return total_lidar_loss
    
    def style_consistency_loss(self, style_sim: torch.Tensor, style_real: torch.Tensor) -> torch.Tensor:
        """风格一致性损失 - 学习目标域的风格"""
        # 使用余弦相似度，让不同域的风格有所区别
        cosine_sim = F.cosine_similarity(style_sim, style_real, dim=1)
        # 我们希望风格有区别但不要完全相反
        return (cosine_sim - 0.5).abs().mean()
    
    def smoothness_loss(self, generated: torch.Tensor) -> torch.Tensor:
        """平滑性损失 - 避免生成的点云有异常的尖锐变化"""
        # 简化版本，避免计算所有点对
        batch_size = generated.shape[0]
        
        # 只计算相邻点的差异
        diff = generated[:, 1:] - generated[:, :-1]  # [B, N-1, 3]
        diff_norm = torch.norm(diff, dim=2)  # [B, N-1]
        
        # 平滑度：相邻点距离的方差应该小
        smooth_loss = diff_norm.std(dim=1).mean()
        
        return smooth_loss
    
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
                t: Optional[torch.Tensor] = None,
                warmup_step: int = 1000,
                global_step: int = 0) -> Dict[str, torch.Tensor]:
        """
        计算总损失（添加LiDAR结构损失）
        """
        losses = {}
        
        # 预热因子
        warmup_factor = min(1.0, global_step / warmup_step) if warmup_step > 0 else 1.0
        
        # 1. 基础Diffusion损失
        diff_loss = self.diffusion_loss(pred_noise, target_noise)
        losses['diffusion'] = diff_loss
        
        # 2. 内容保持损失
        if content_original is not None and content_from_noisy is not None:
            content_loss = self.content_preservation_loss(content_original, content_from_noisy)
            losses['content'] = content_loss
        else:
            losses['content'] = torch.tensor(0.0, device=pred_noise.device)
        
        # 3. 形状保持损失
        if generated_points is not None and original_points is not None:
            shape_loss = self.shape_preservation_loss(generated_points, original_points)
            losses['shape'] = shape_loss
            
            # 4. 局部结构损失
            local_loss = self.local_structure_loss(generated_points, original_points)
            losses['local'] = local_loss
            
            # 5. 平滑性损失
            smooth_loss = self.smoothness_loss(generated_points)
            losses['smooth'] = smooth_loss
            
            # 6. LiDAR结构损失（新增）
            lidar_loss = self.lidar_structure_loss(generated_points, original_points)
            losses['lidar_structure'] = lidar_loss
        else:
            losses['shape'] = torch.tensor(0.0, device=pred_noise.device)
            losses['local'] = torch.tensor(0.0, device=pred_noise.device)
            losses['smooth'] = torch.tensor(0.0, device=pred_noise.device)
            losses['lidar_structure'] = torch.tensor(0.0, device=pred_noise.device)
        
        # 7. 风格损失
        if style_source is not None and style_target is not None:
            style_loss = self.style_consistency_loss(style_source, style_target)
            losses['style'] = style_loss
        else:
            losses['style'] = torch.tensor(0.0, device=pred_noise.device)
        
        # 更新EMA
        self.update_loss_ema(losses)
        
        # 获取自适应权重
        weights = self.get_adaptive_weights()
        
        # 计算总损失，应用预热和自适应权重
        total_loss = weights['diffusion'] * losses['diffusion']
        
        # 其他损失应用预热
        total_loss += weights['content'] * losses['content'] * warmup_factor
        total_loss += weights['shape'] * losses['shape'] * warmup_factor
        total_loss += weights['local'] * losses['local'] * warmup_factor
        total_loss += weights['smooth'] * losses['smooth'] * warmup_factor
        total_loss += weights['lidar_structure'] * losses['lidar_structure'] * warmup_factor
        total_loss += weights['style'] * losses['style'] * warmup_factor
        
        losses['total'] = total_loss
        
        # 添加权重信息用于监控
        if global_step % 100 == 0:
            for k, v in weights.items():
                losses[f'weight_{k}'] = torch.tensor(v)
        
        return losses