import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算Chamfer距离
    Args:
        pred: 预测点云 [B, N, 3]
        target: 目标点云 [B, M, 3]
    Returns:
        Chamfer距离的两个方向
    """
    # 计算距离矩阵 [B, N, M]
    dist_matrix = torch.cdist(pred, target, p=2)
    
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
    # 简化的EMD计算，使用最小二分图匹配的近似
    batch_size, num_points, _ = pred.shape
    
    # 计算成本矩阵
    cost_matrix = torch.cdist(pred, target, p=2)  # [B, N, N]
    
    # 使用匈牙利算法的近似 - 贪心匹配
    emd_loss = []
    for b in range(batch_size):
        cost = cost_matrix[b]  # [N, N]
        
        # 简单的最小权重匹配
        total_cost = 0
        used_target = set()
        
        for i in range(num_points):
            min_cost = float('inf')
            best_j = -1
            
            for j in range(num_points):
                if j not in used_target and cost[i, j] < min_cost:
                    min_cost = cost[i, j]
                    best_j = j
            
            if best_j != -1:
                total_cost += min_cost
                used_target.add(best_j)
        
        emd_loss.append(total_cost / num_points)
    
    return torch.tensor(emd_loss, device=pred.device, dtype=pred.dtype)


class ChamferLoss(nn.Module):
    """Chamfer距离损失"""
    
    def __init__(self, use_sqrt: bool = False):
        super(ChamferLoss, self).__init__()
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
        super(EMDLoss, self).__init__()
    
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


class PerceptualLoss(nn.Module):
    """感知损失，基于PointNet特征"""
    
    def __init__(self, feature_extractor: nn.Module):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        
        # 冻结特征提取器
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失
        Args:
            pred: 预测点云 [B, N, 3]
            target: 目标点云 [B, N, 3]
        Returns:
            感知损失
        """
        # 提取特征
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # 计算特征距离
        if isinstance(pred_features, tuple):
            # 如果返回多个特征层
            loss = 0
            for pred_feat, target_feat in zip(pred_features, target_features):
                loss += F.mse_loss(pred_feat, target_feat)
            return loss
        else:
            return F.mse_loss(pred_features, target_features)


class AdversarialLoss(nn.Module):
    """对抗损失"""
    
    def __init__(self, loss_type: str = 'lsgan'):
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'wgangp':
            self.criterion = None
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        计算对抗损失
        Args:
            pred: 判别器预测 [B, 1]
            target_is_real: 目标是否为真实数据
        Returns:
            对抗损失
        """
        if self.loss_type == 'wgangp':
            # Wasserstein GAN with Gradient Penalty
            if target_is_real:
                return -torch.mean(pred)
            else:
                return torch.mean(pred)
        else:
            # LSGAN或Vanilla GAN
            if target_is_real:
                target = torch.ones_like(pred)
            else:
                target = torch.zeros_like(pred)
            
            return self.criterion(pred, target)


class GradientPenalty(nn.Module):
    """梯度惩罚（用于WGAN-GP）"""
    
    def __init__(self, lambda_gp: float = 10.0):
        super(GradientPenalty, self).__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, discriminator: nn.Module, real_data: torch.Tensor, 
                fake_data: torch.Tensor) -> torch.Tensor:
        """
        计算梯度惩罚
        Args:
            discriminator: 判别器模型
            real_data: 真实数据 [B, N, 3]
            fake_data: 生成数据 [B, N, 3]
        Returns:
            梯度惩罚损失
        """
        batch_size = real_data.size(0)
        
        # 随机插值
        alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # 计算判别器输出
        disc_interpolated = discriminator(interpolated)[0]
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度范数
        gradients = gradients.reshape(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        
        # 梯度惩罚
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return self.lambda_gp * gradient_penalty


class FeatureMatchingLoss(nn.Module):
    """特征匹配损失"""
    
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
    
    def forward(self, fake_features: List[torch.Tensor], 
                real_features: List[torch.Tensor]) -> torch.Tensor:
        """
        计算特征匹配损失
        Args:
            fake_features: 生成数据的特征列表
            real_features: 真实数据的特征列表
        Returns:
            特征匹配损失
        """
        loss = 0
        for fake_feat, real_feat in zip(fake_features, real_features):
            loss += F.l1_loss(fake_feat.mean(0), real_feat.mean(0).detach())
        
        return loss


class TotalVariationLoss(nn.Module):
    """总变分损失，用于平滑性正则化"""
    
    def __init__(self, k: int = 8):
        super(TotalVariationLoss, self).__init__()
        self.k = k
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        计算总变分损失
        Args:
            points: 点云 [B, N, 3]
        Returns:
            总变分损失
        """
        batch_size, num_points, _ = points.shape
        
        # 计算K近邻
        dist_matrix = torch.cdist(points, points, p=2)
        _, knn_idx = torch.topk(dist_matrix, k=self.k + 1, dim=-1, largest=False)
        knn_idx = knn_idx[:, :, 1:]  # 排除自身
        
        # 计算邻居点的变化
        total_variation = 0
        for b in range(batch_size):
            for i in range(num_points):
                center = points[b, i:i+1, :]  # [1, 3]
                neighbors = points[b, knn_idx[b, i], :]  # [k, 3]
                variations = torch.sum((neighbors - center) ** 2, dim=1)
                total_variation += torch.mean(variations)
        
        return total_variation / (batch_size * num_points)


class StyleTransferLoss(nn.Module):
    """风格迁移总损失"""
    
    def __init__(self, lambda_recon: float = 10.0,
                 lambda_adv: float = 1.0,
                 lambda_cycle: float = 5.0,
                 lambda_identity: float = 2.0,
                 lambda_perceptual: float = 1.0,
                 lambda_feature: float = 1.0):
        super(StyleTransferLoss, self).__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_adv = lambda_adv
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_perceptual = lambda_perceptual
        self.lambda_feature = lambda_feature
        
        # 损失函数
        self.chamfer_loss = ChamferLoss()
        self.adversarial_loss = AdversarialLoss('lsgan')
        self.feature_matching_loss = FeatureMatchingLoss()
        
    def generator_loss(self, fake_output: torch.Tensor,
                      fake_features: List[torch.Tensor],
                      real_features: List[torch.Tensor],
                      cycled_data: Optional[torch.Tensor] = None,
                      original_data: Optional[torch.Tensor] = None,
                      identity_data: Optional[torch.Tensor] = None,
                      target_data: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        计算生成器总损失
        Args:
            fake_output: 判别器对生成数据的输出
            fake_features: 生成数据的特征
            real_features: 真实数据的特征
            cycled_data: 循环重建的数据
            original_data: 原始数据
            identity_data: 身份映射的数据
            target_data: 目标数据
        Returns:
            总损失和损失字典
        """
        losses = {}
        
        # 对抗损失
        adv_loss = self.adversarial_loss(fake_output, True)
        losses['adversarial'] = adv_loss
        
        # 特征匹配损失
        if fake_features and real_features:
            fm_loss = self.feature_matching_loss(fake_features, real_features)
            losses['feature_matching'] = fm_loss
        else:
            fm_loss = 0
        
        # 循环一致性损失
        if cycled_data is not None and original_data is not None:
            cycle_loss = self.chamfer_loss(cycled_data, original_data)
            losses['cycle'] = cycle_loss
        else:
            cycle_loss = 0
        
        # 身份损失
        if identity_data is not None and target_data is not None:
            identity_loss = self.chamfer_loss(identity_data, target_data)
            losses['identity'] = identity_loss
        else:
            identity_loss = 0
        
        # 总损失
        total_loss = (self.lambda_adv * adv_loss +
                     self.lambda_feature * fm_loss +
                     self.lambda_cycle * cycle_loss +
                     self.lambda_identity * identity_loss)
        
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def discriminator_loss(self, real_output: torch.Tensor,
                          fake_output: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        计算判别器损失
        Args:
            real_output: 判别器对真实数据的输出
            fake_output: 判别器对生成数据的输出
        Returns:
            总损失和损失字典
        """
        # 真实数据损失
        real_loss = self.adversarial_loss(real_output, True)
        
        # 生成数据损失
        fake_loss = self.adversarial_loss(fake_output, False)
        
        # 总损失
        total_loss = (real_loss + fake_loss) * 0.5
        
        losses = {
            'real': real_loss,
            'fake': fake_loss,
            'total': total_loss
        }
        
        return total_loss, losses


def test_losses():
    """测试损失函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 4
    num_points = 1024
    pred = torch.randn(batch_size, num_points, 3).to(device)
    target = torch.randn(batch_size, num_points, 3).to(device)
    
    # 测试Chamfer损失
    chamfer_loss = ChamferLoss()
    cd_loss = chamfer_loss(pred, target)
    print(f"Chamfer Distance Loss: {cd_loss.item():.4f}")
    
    # 测试EMD损失
    emd_loss = EMDLoss()
    emd_value = emd_loss(pred, target)
    print(f"EMD Loss: {emd_value.item():.4f}")
    
    # 测试对抗损失
    adv_loss = AdversarialLoss('lsgan')
    fake_scores = torch.randn(batch_size, 1).to(device)
    
    gen_loss = adv_loss(fake_scores, True)
    disc_loss = adv_loss(fake_scores, False)
    print(f"Adversarial Loss (Gen): {gen_loss.item():.4f}")
    print(f"Adversarial Loss (Disc): {disc_loss.item():.4f}")
    
    # 测试总变分损失
    tv_loss = TotalVariationLoss()
    tv_value = tv_loss(pred)
    print(f"Total Variation Loss: {tv_value.item():.4f}")


if __name__ == "__main__":
    test_losses()