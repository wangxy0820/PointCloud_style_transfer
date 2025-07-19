import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .pointnet2 import EdgeConv, get_graph_feature


class SpectralNorm(nn.Module):
    """谱归一化"""
    
    def __init__(self, module: nn.Module, name: str = 'weight', power_iterations: int = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.reshape(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.reshape(height,-1).data, v.data))

        sigma = u.dot(w.reshape(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.reshape(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def spectral_norm(module, name='weight', power_iterations=1):
    """应用谱归一化"""
    fn = SpectralNorm(module, name, power_iterations)
    return fn


class PointCloudDiscriminator(nn.Module):
    """点云判别器"""
    
    def __init__(self, input_channels: int = 3,
                 feature_channels: List[int] = [64, 128, 256, 512],
                 k: int = 20,
                 use_spectral_norm: bool = True):
        super(PointCloudDiscriminator, self).__init__()
        
        self.k = k
        
        # 特征提取层
        self.edge_convs = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in feature_channels:
            conv = EdgeConv(in_channels, out_channels, k)
            if use_spectral_norm:
                # 对卷积层应用谱归一化
                conv.conv[0] = spectral_norm(conv.conv[0])
            self.edge_convs.append(conv)
            in_channels = out_channels
        
        # 全局特征提取
        self.global_conv = nn.Conv1d(sum(feature_channels), 1024, 1)
        if use_spectral_norm:
            self.global_conv = spectral_norm(self.global_conv)
        
        self.global_bn = nn.BatchNorm1d(1024)
        
        # 分类头
        classifier_layers = [
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        ]
        
        if use_spectral_norm:
            classifier_layers[0] = spectral_norm(classifier_layers[0])
            classifier_layers[3] = spectral_norm(classifier_layers[3])
            classifier_layers[6] = spectral_norm(classifier_layers[6])
        
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        判别器前向传播
        Args:
            x: 输入点云 [B, N, 3]
        Returns:
            判别结果 [B, 1] 和特征 [B, 1024]
        """
        # 转换为 [B, 3, N] 格式
        x = x.transpose(2, 1).contiguous()
        
        # 提取多层次特征
        features = []
        for edge_conv in self.edge_convs:
            x = edge_conv(x)
            features.append(x)
        
        # 连接所有特征
        x = torch.cat(features, dim=1)  # [B, sum(feature_channels), N]
        
        # 全局特征提取
        global_feature = F.leaky_relu(self.global_bn(self.global_conv(x)), 0.2)
        global_feature = F.adaptive_max_pool1d(global_feature, 1)  # [B, 1024, 1]
        global_feature = global_feature.reshape(global_feature.size(0), -1)  # [B, 1024]
        
        # 分类
        output = self.classifier(global_feature)  # [B, 1]
        
        return output, global_feature


class MultiScaleDiscriminator(nn.Module):
    """多尺度判别器"""
    
    def __init__(self, input_channels: int = 3,
                 scales: List[int] = [8192, 4096, 2048],
                 feature_channels: List[int] = [64, 128, 256, 512]):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.scales = scales
        self.discriminators = nn.ModuleList()
        
        # 为每个尺度创建判别器
        for scale in scales:
            disc = PointCloudDiscriminator(
                input_channels=input_channels,
                feature_channels=feature_channels
            )
            self.discriminators.append(disc)
    
    def downsample(self, points: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        下采样点云
        Args:
            points: 输入点云 [B, N, 3]
            target_size: 目标点数
        Returns:
            下采样后的点云 [B, target_size, 3]
        """
        B, N, C = points.shape
        if N <= target_size:
            return points
        
        # 随机采样
        indices = torch.randperm(N)[:target_size]
        return points[:, indices, :]
    
    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        多尺度判别
        Args:
            x: 输入点云 [B, N, 3]
        Returns:
            每个尺度的判别结果和特征列表
        """
        results = []
        
        for i, (discriminator, scale) in enumerate(zip(self.discriminators, self.scales)):
            # 下采样到对应尺度
            if i == 0:
                scaled_x = x  # 第一个尺度使用原始输入
            else:
                scaled_x = self.downsample(x, scale)
            
            # 判别
            output, features = discriminator(scaled_x)
            results.append((output, features))
        
        return results


class PatchDiscriminator(nn.Module):
    """补丁判别器"""
    
    def __init__(self, input_channels: int = 3,
                 feature_channels: List[int] = [64, 128, 256],
                 patch_size: int = 1024):
        super(PatchDiscriminator, self).__init__()
        
        self.patch_size = patch_size
        
        # 局部特征提取
        self.local_convs = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in feature_channels:
            self.local_convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2)
            ))
            in_channels = out_channels
        
        # 补丁分类器
        self.patch_classifier = nn.Sequential(
            nn.Conv1d(feature_channels[-1], 128, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 1, 1)
        )
        
    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取补丁
        Args:
            x: 输入点云 [B, N, 3]
        Returns:
            补丁 [B, num_patches, patch_size, 3]
        """
        B, N, C = x.shape
        num_patches = N // self.patch_size
        
        # 重新排列为补丁
        patches = x[:, :num_patches * self.patch_size, :].reshape(
            B, num_patches, self.patch_size, C
        )
        
        return patches
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        补丁判别
        Args:
            x: 输入点云 [B, N, 3]
        Returns:
            每个补丁的判别结果 [B, num_patches]
        """
        # 提取补丁
        patches = self.extract_patches(x)  # [B, num_patches, patch_size, 3]
        B, num_patches, patch_size, C = patches.shape
        
        # 重新整理为 [B*num_patches, patch_size, 3]
        patches = patches.reshape(B * num_patches, patch_size, C)
        
        # 转换为 [B*num_patches, 3, patch_size]
        patches = patches.transpose(2, 1).contiguous()
        
        # 特征提取
        features = patches
        for conv in self.local_convs:
            features = conv(features)
        
        # 补丁分类
        patch_outputs = self.patch_classifier(features)  # [B*num_patches, 1, patch_size]
        patch_outputs = F.adaptive_avg_pool1d(patch_outputs, 1)  # [B*num_patches, 1, 1]
        patch_outputs = patch_outputs.reshape(B, num_patches)  # [B, num_patches]
        
        return patch_outputs


class HybridDiscriminator(nn.Module):
    """混合判别器，结合全局和局部判别"""
    
    def __init__(self, input_channels: int = 3,
                 feature_channels: List[int] = [64, 128, 256, 512],
                 patch_size: int = 1024):
        super(HybridDiscriminator, self).__init__()
        
        # 全局判别器
        self.global_discriminator = PointCloudDiscriminator(
            input_channels, feature_channels
        )
        
        # 补丁判别器
        self.patch_discriminator = PatchDiscriminator(
            input_channels, feature_channels[:3], patch_size
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        混合判别
        Args:
            x: 输入点云 [B, N, 3]
        Returns:
            判别结果, 全局特征
        """
        # 全局判别
        global_output, global_features = self.global_discriminator(x)
    
        # 补丁判别
        patch_outputs = self.patch_discriminator(x)
    
        # 将补丁判别结果整合到全局输出中
        # 使用平均池化将补丁输出转换为单一值
        patch_score = patch_outputs.mean(dim=1, keepdim=True)  # [B, 1]
    
        # 组合全局和补丁判别结果
        combined_output = 0.5 * global_output + 0.5 * patch_score
    
        return combined_output, global_features


def test_discriminator():
    """测试判别器模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 4
    num_points = 8192
    x = torch.randn(batch_size, num_points, 3).to(device)
    
    # 测试基础判别器
    print("Testing PointCloudDiscriminator...")
    discriminator = PointCloudDiscriminator().to(device)
    output, features = discriminator(x)
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    
    # 测试多尺度判别器
    print("\nTesting MultiScaleDiscriminator...")
    multi_disc = MultiScaleDiscriminator().to(device)
    results = multi_disc(x)
    for i, (out, feat) in enumerate(results):
        print(f"Scale {i}: Output shape: {out.shape}, Features shape: {feat.shape}")
    
    # 测试补丁判别器
    print("\nTesting PatchDiscriminator...")
    patch_disc = PatchDiscriminator().to(device)
    patch_output = patch_disc(x)
    print(f"Patch output shape: {patch_output.shape}")
    
    # 测试混合判别器
    print("\nTesting HybridDiscriminator...")
    hybrid_disc = HybridDiscriminator().to(device)
    global_out, global_feat, patch_out = hybrid_disc(x)
    print(f"Global output shape: {global_out.shape}")
    print(f"Global features shape: {global_feat.shape}")
    print(f"Patch output shape: {patch_out.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nDiscriminator parameters: {total_params:,}")


if __name__ == "__main__":
    test_discriminator()