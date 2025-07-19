import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .pointnet2 import PointNet2Encoder, PointNet2Decoder


class StyleEncoder(nn.Module):
    """风格编码器"""
    
    def __init__(self, input_dim: int = 512, style_dim: int = 256):
        super(StyleEncoder, self).__init__()
        
        self.style_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, style_dim)
        )
        
    def forward(self, global_feature: torch.Tensor) -> torch.Tensor:
        """
        提取风格特征
        Args:
            global_feature: 全局特征 [B, input_dim]
        Returns:
            风格特征 [B, style_dim]
        """
        return self.style_net(global_feature)


class AdaptiveInstanceNorm(nn.Module):
    """自适应实例归一化"""
    
    def __init__(self, num_features: int, style_dim: int):
        super(AdaptiveInstanceNorm, self).__init__()
        self.num_features = num_features
        self.style_dim = style_dim
        
        # 风格变换网络
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_shift = nn.Linear(style_dim, num_features)
        
        # 初始化
        nn.init.normal_(self.style_scale.weight, 1.0, 0.02)
        nn.init.constant_(self.style_scale.bias, 0)
        nn.init.constant_(self.style_shift.weight, 0)
        nn.init.constant_(self.style_shift.bias, 0)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        应用自适应实例归一化
        Args:
            x: 输入特征 [B, C, N]
            style: 风格特征 [B, style_dim]
        Returns:
            归一化后的特征 [B, C, N]
        """
        # 实例归一化
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        normalized = (x - mean) / std
        
        # 风格变换
        scale = self.style_scale(style).unsqueeze(2)  # [B, C, 1]
        shift = self.style_shift(style).unsqueeze(2)  # [B, C, 1]
        
        return scale * normalized + shift


class StyleTransferBlock(nn.Module):
    """风格迁移块"""
    
    def __init__(self, in_channels: int, out_channels: int, style_dim: int):
        super(StyleTransferBlock, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.adain = AdaptiveInstanceNorm(out_channels, style_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        风格迁移前向传播
        Args:
            x: 输入特征 [B, in_channels, N]
            style: 风格特征 [B, style_dim]
        Returns:
            输出特征 [B, out_channels, N]
        """
        x = self.conv(x)
        x = self.adain(x, style)
        x = self.activation(x)
        return x


class AttentionModule(nn.Module):
    """注意力模块"""
    
    def __init__(self, channels: int):
        super(AttentionModule, self).__init__()
        
        self.query_conv = nn.Conv1d(channels, channels // 8, 1)
        self.key_conv = nn.Conv1d(channels, channels // 8, 1)
        self.value_conv = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        自注意力机制
        Args:
            x: 输入特征 [B, C, N]
        Returns:
            注意力增强的特征 [B, C, N]
        """
        B, C, N = x.size()
        
        # 计算查询、键、值
        query = self.query_conv(x).reshape(B, -1, N).permute(0, 2, 1)  # [B, N, C//8]
        key = self.key_conv(x).reshape(B, -1, N)                       # [B, C//8, N]
        value = self.value_conv(x).reshape(B, -1, N)                   # [B, C, N]
        
        # 计算注意力权重
        attention = torch.bmm(query, key)  # [B, N, N]
        attention = F.softmax(attention, dim=-1)
        
        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, N]
        out = self.gamma * out + x
        
        return out


class PointCloudGenerator(nn.Module):
    """点云生成器"""
    
    def __init__(self, input_channels: int = 3,
                 feature_channels: List[int] = [64, 128, 256, 512],
                 style_dim: int = 256,
                 latent_dim: int = 512,
                 num_points: int = 8192):
        super(PointCloudGenerator, self).__init__()
    # def __init__(self, input_channels: int = 3,
    #              feature_channels: List[int] = [32, 64, 128, 256],
    #              style_dim: int = 128,
    #              latent_dim: int = 256,
    #              num_points: int = 2048):
    #     super(PointCloudGenerator, self).__init__()
        
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        
        # 内容编码器
        self.content_encoder = PointNet2Encoder(
            input_channels, feature_channels, latent_dim
        )
        
        # 风格编码器
        self.style_encoder = StyleEncoder(latent_dim, style_dim)
        
        # 风格迁移网络
        self.style_transfer_layers = nn.ModuleList([
            StyleTransferBlock(latent_dim, 512, style_dim),
            StyleTransferBlock(512, 256, style_dim),
            StyleTransferBlock(256, 128, style_dim),
        ])
        
        # self.style_transfer_layers = nn.ModuleList([
        #     StyleTransferBlock(latent_dim, 256, style_dim),
        #     StyleTransferBlock(256, 128, style_dim),
        # ])
        
        # 注意力模块
        self.attention = AttentionModule(128)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, input_channels, 1),
            nn.Tanh()
        )
        
        # 全局特征解码器
        self.global_decoder = nn.Sequential(
            nn.Linear(latent_dim + style_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)  # 不硬编码点数
        )
        
        # 全局特征到3D点的投影层
        self.global_to_3d = nn.Linear(256, input_channels)
        
    def forward(self, content_points: torch.Tensor,
                style_points: torch.Tensor) -> torch.Tensor:
        """
        生成器前向传播
        Args:
            content_points: 内容点云 [B, N, 3]
            style_points: 风格参考点云 [B, N, 3]
        Returns:
            生成的点云 [B, N, 3]
        """
        # 编码内容特征
        content_local, content_global = self.content_encoder(content_points)
        
        # 编码风格特征
        _, style_global = self.content_encoder(style_points)
        style_feature = self.style_encoder(style_global)
        
        # 风格迁移
        x = content_local
        for layer in self.style_transfer_layers:
            x = layer(x, style_feature)
        
        # 应用注意力
        x = self.attention(x)
        
        # 全局解码
        global_input = torch.cat([content_global, style_feature], dim=1)
        global_feature = self.global_decoder(global_input)  # [B, 256]
        
        # 动态扩展全局特征到匹配点数
        num_points = content_local.size(2)
        
        # 将全局特征扩展并投影到3D空间
        global_expanded = global_feature.unsqueeze(1).expand(-1, num_points, -1)  # [B, N, 256]
        global_points = self.global_to_3d(global_expanded)  # [B, N, 3]
        
        local_output = self.decoder(x).transpose(2, 1)  # [B, N, 3]
        output = 0.7 * local_output + 0.3 * global_points
        
        return output
    
    def transfer_style(self, content_points: torch.Tensor,
                      style_feature: torch.Tensor) -> torch.Tensor:
        """
        使用给定的风格特征进行风格迁移
        Args:
            content_points: 内容点云 [B, N, 3]
            style_feature: 风格特征 [B, style_dim]
        Returns:
            风格迁移后的点云 [B, N, 3]
        """
        # 编码内容特征
        content_local, content_global = self.content_encoder(content_points)
        
        # 风格迁移
        x = content_local
        for layer in self.style_transfer_layers:
            x = layer(x, style_feature)
        
        # 应用注意力
        x = self.attention(x)
        
        # 局部解码
        local_output = self.decoder(x).transpose(2, 1)  # [B, N, 3]
        
        # 全局解码
        global_input = torch.cat([content_global, style_feature], dim=1)
        global_feature = self.global_decoder(global_input)  # [B, 256]
        
        # 动态扩展全局特征到匹配点数
        num_points = content_points.size(1)
        global_expanded = global_feature.unsqueeze(1).expand(-1, num_points, -1)  # [B, N, 256]
        global_points = self.global_to_3d(global_expanded)  # [B, N, 3]
        
        # 融合输出
        output = 0.7 * local_output + 0.3 * global_points
        
        return output


class CycleConsistentGenerator(nn.Module):
    """循环一致性生成器"""
    
    def __init__(self, input_channels: int = 3,
                 feature_channels: List[int] = [64, 128, 256, 512],
                 style_dim: int = 256,
                 latent_dim: int = 512,
                 num_points: int = 8192):
        super(CycleConsistentGenerator, self).__init__()
        
        # Sim2Real生成器
        self.sim2real = PointCloudGenerator(
            input_channels, feature_channels, style_dim, latent_dim, num_points
        )
        
        # Real2Sim生成器
        self.real2sim = PointCloudGenerator(
            input_channels, feature_channels, style_dim, latent_dim, num_points
        )
        
    def forward(self, sim_points: torch.Tensor, real_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        循环一致性前向传播
        Args:
            sim_points: 仿真点云 [B, N, 3]
            real_points: 真实点云 [B, N, 3]
        Returns:
            fake_real: sim->real生成的点云
            fake_sim: real->sim生成的点云
        """
        fake_real = self.sim2real(sim_points, real_points)
        fake_sim = self.real2sim(real_points, sim_points)
        
        return fake_real, fake_sim
    
    def cycle_forward(self, sim_points: torch.Tensor, real_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整的循环前向传播
        Args:
            sim_points: 仿真点云 [B, N, 3]
            real_points: 真实点云 [B, N, 3]
        Returns:
            cycled_sim: sim->real->sim的循环结果
            cycled_real: real->sim->real的循环结果
        """
        # 第一步转换
        fake_real = self.sim2real(sim_points, real_points)
        fake_sim = self.real2sim(real_points, sim_points)
        
        # 循环转换
        cycled_sim = self.real2sim(fake_real, sim_points)
        cycled_real = self.sim2real(fake_sim, real_points)
        
        return cycled_sim, cycled_real


def test_generator():
    """测试生成器模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    generator = PointCloudGenerator(
        input_channels=3,
        feature_channels=[64, 128, 256, 512],
        style_dim=256,
        latent_dim=512,
        num_points=8192
    ).to(device)
    
    # 创建测试数据
    batch_size = 4
    num_points = 8192
    content_points = torch.randn(batch_size, num_points, 3).to(device)
    style_points = torch.randn(batch_size, num_points, 3).to(device)
    
    # 前向传播
    generated_points = generator(content_points, style_points)
    
    print(f"Content points shape: {content_points.shape}")
    print(f"Style points shape: {style_points.shape}")
    print(f"Generated points shape: {generated_points.shape}")
    
    # 测试循环一致性生成器
    cycle_generator = CycleConsistentGenerator().to(device)
    fake_real, fake_sim = cycle_generator(content_points, style_points)
    
    print(f"Fake real shape: {fake_real.shape}")
    print(f"Fake sim shape: {fake_sim.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {total_params:,}")


if __name__ == "__main__":
    test_generator()