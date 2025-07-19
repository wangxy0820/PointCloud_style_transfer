"""
配置文件示例
包含不同训练场景的配置模板
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple
from .config import Config


@dataclass
class QuickTestConfig(Config):
    """快速测试配置 - 用于验证代码运行"""
    
    # 小规模数据
    chunk_size: int = 1024
    num_chunks: int = 4
    batch_size: int = 2
    
    # 短训练时间
    num_epochs: int = 5
    log_interval: int = 10
    save_interval: int = 2
    eval_interval: int = 2
    
    # 简化模型
    pointnet_channels: List[int] = (32, 64, 128)
    latent_dim: int = 128
    generator_dim: int = 64
    
    # 少量验证样本
    vis_samples: int = 2


@dataclass
class StandardTrainingConfig(Config):
    """标准训练配置 - 推荐的生产配置"""
    
    # 标准参数 - 继承默认值
    #chunk_size: int = 8192
    chunk_size: int = 4096
    batch_size: int = 2
    num_epochs: int = 200
    
    # 平衡的损失权重
    lambda_recon: float = 10.0
    lambda_adv: float = 1.0
    lambda_cycle: float = 5.0
    lambda_identity: float = 2.0
    
    # 数据增强
    use_rotation: bool = True
    use_jitter: bool = True
    use_scaling: bool = True
    rotation_range: float = 0.1
    jitter_std: float = 0.01
    scaling_range: Tuple[float, float] = (0.9, 1.1)


@dataclass
class HighQualityConfig(Config):
    """高质量训练配置 - 追求最佳效果"""
    
    # 大模型
    chunk_size: int = 10240
    pointnet_channels: List[int] = (128, 256, 512, 1024)
    latent_dim: int = 1024
    generator_dim: int = 512
    
    # 更多训练轮数
    num_epochs: int = 500
    warmup_epochs: int = 20
    
    # 更高质量要求的损失权重
    lambda_recon: float = 15.0
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    
    # 更小的学习率
    learning_rate_g: float = 0.0001
    learning_rate_d: float = 0.00005
    
    # 更多验证样本
    vis_samples: int = 10


@dataclass
class FastTrainingConfig(Config):
    """快速训练配置 - 优先训练速度"""
    
    # 更大的批次
    batch_size: int = 16
    
    # 更小的模型
    pointnet_channels: List[int] = (64, 128, 256)
    latent_dim: int = 256
    generator_dim: int = 128
    
    # 更少的训练步骤
    discriminator_steps: int = 2  # 减少判别器训练频率
    
    # 更高的学习率
    learning_rate_g: float = 0.0005
    learning_rate_d: float = 0.0002
    
    # 更少的评估
    eval_interval: int = 10
    log_interval: int = 50


@dataclass
class MemoryEfficientConfig(Config):
    """内存高效配置 - 适用于GPU内存有限的情况"""
    
    # 小批次和小块
    batch_size: int = 2
    chunk_size: int = 4096
    
    # 更小的模型
    pointnet_channels: List[int] = (32, 64, 128, 256)
    latent_dim: int = 256
    generator_dim: int = 128
    
    # 减少工作进程
    num_workers: int = 2
    pin_memory: bool = False


@dataclass
class RobustTrainingConfig(Config):
    """鲁棒性训练配置 - 重视训练稳定性"""
    
    # 保守的学习率
    learning_rate_g: float = 0.0001
    learning_rate_d: float = 0.00005
    
    # 更长的预热
    warmup_epochs: int = 50
    
    # 平衡的损失权重
    lambda_recon: float = 20.0  # 更重视重建质量
    lambda_adv: float = 0.5     # 减少对抗训练的激进程度
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    
    # 更频繁的保存
    save_interval: int = 5
    eval_interval: int = 3
    
    # 梯度裁剪（需要在训练器中实现）
    max_grad_norm: float = 1.0


@dataclass
class LargeScaleConfig(Config):
    """大规模数据配置 - 处理大量数据"""
    
    # 大批次
    batch_size: int = 32
    
    # 更多数据工作进程
    num_workers: int = 8
    pin_memory: bool = True
    
    # 更频繁的日志（因为每个epoch时间长）
    log_interval: int = 50
    
    # 更少的可视化（节省时间）
    vis_samples: int = 2
    
    # 更大的模型容量
    pointnet_channels: List[int] = (128, 256, 512, 1024)
    latent_dim: int = 1024
    generator_dim: int = 512


@dataclass 
class ExperimentalConfig(Config):
    """实验性配置 - 用于尝试新想法"""
    
    # 不对称的损失权重
    lambda_recon: float = 5.0
    lambda_adv: float = 2.0
    lambda_cycle: float = 15.0  # 强调循环一致性
    lambda_identity: float = 0.5
    
    # 激进的数据增强
    rotation_range: float = 0.2
    jitter_std: float = 0.02
    scaling_range: Tuple[float, float] = (0.8, 1.2)
    
    # 不同的训练策略
    discriminator_steps: int = 3  # 更多判别器训练
    
    # 更复杂的模型
    pointnet_channels: List[int] = (64, 128, 256, 512, 1024)


def get_config_by_name(config_name: str) -> Config:
    """
    根据名称获取配置
    Args:
        config_name: 配置名称
    Returns:
        配置对象
    """
    config_map = {
        'quick_test': QuickTestConfig(),
        'standard': StandardTrainingConfig(),
        'high_quality': HighQualityConfig(),
        'fast_training': FastTrainingConfig(),
        'memory_efficient': MemoryEfficientConfig(),
        'robust': RobustTrainingConfig(),
        'large_scale': LargeScaleConfig(),
        'experimental': ExperimentalConfig(),
    }
    
    if config_name not in config_map:
        available_configs = list(config_map.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available configs: {available_configs}")
    
    return config_map[config_name]


def print_config_comparison():
    """打印所有配置的对比"""
    configs = {
        'QuickTest': QuickTestConfig(),
        'Standard': StandardTrainingConfig(),
        'HighQuality': HighQualityConfig(),
        'FastTraining': FastTrainingConfig(),
        'MemoryEfficient': MemoryEfficientConfig(),
        'Robust': RobustTrainingConfig(),
        'LargeScale': LargeScaleConfig(),
        'Experimental': ExperimentalConfig(),
    }
    
    print("Configuration Comparison")
    print("=" * 80)
    
    # 打印表头
    print(f"{'Config':<15} {'Batch':<6} {'Chunk':<6} {'Epochs':<7} {'LatentDim':<10} {'LR_G':<8}")
    print("-" * 80)
    
    # 打印每个配置的关键参数
    for name, config in configs.items():
        print(f"{name:<15} {config.batch_size:<6} {config.chunk_size:<6} "
              f"{config.num_epochs:<7} {config.latent_dim:<10} {config.learning_rate_g:<8.1e}")
    
    print("\nLoss Weight Comparison")
    print("-" * 80)
    print(f"{'Config':<15} {'Recon':<8} {'Adv':<6} {'Cycle':<8} {'Identity':<8}")
    print("-" * 80)
    
    for name, config in configs.items():
        print(f"{name:<15} {config.lambda_recon:<8.1f} {config.lambda_adv:<6.1f} "
              f"{config.lambda_cycle:<8.1f} {config.lambda_identity:<8.1f}")


if __name__ == "__main__":
    # 演示配置使用
    print("Point Cloud Style Transfer - Configuration Examples")
    print("=" * 60)
    
    # 显示配置对比
    print_config_comparison()
    
    # 演示如何使用配置
    print("\n\nExample Usage:")
    print("=" * 60)
    
    # 获取标准配置
    config = get_config_by_name('standard')
    print(f"Standard config chunk size: {config.chunk_size}")
    print(f"Standard config batch size: {config.batch_size}")
    
    # 获取快速测试配置
    test_config = get_config_by_name('quick_test')
    print(f"Quick test config epochs: {test_config.num_epochs}")
    print(f"Quick test config latent dim: {test_config.latent_dim}")