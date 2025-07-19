import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Config:
    """Configuration class for point cloud style transfer project"""
    
    # 数据配置
    data_root: str = "datasets"
    sim_data_dir: str = "datasets/simulation"
    real_data_dir: str = "datasets/real_world"
    processed_data_dir: str = "datasets/processed"
    
    # 点云参数
    max_points: int = 120000  # 原始点云点数
    chunk_size: int = 4096    # 分块大小
    num_chunks: int = 15      # 分块数量 (120000 / 8192 ≈ 15)
    input_dim: int = 3        # 点云维度 (x, y, z)
    
    # 模型参数
    pointnet_channels: List[int] = (64, 128, 256, 512)
    generator_dim: int = 512
    discriminator_dim: int = 256
    latent_dim: int = 256
    # pointnet_channels: List[int] = (32, 64, 128)
    # generator_dim: int = 128
    # discriminator_dim: int = 64
    # latent_dim: int = 128
    
    # 训练参数
    batch_size: int = 2
    num_epochs: int = 40
    learning_rate_g: float = 0.0002  # 生成器学习率
    learning_rate_d: float = 0.0001  # 判别器学习率
    beta1: float = 0.5
    beta2: float = 0.999
    
    # 损失权重
    lambda_recon: float = 10.0    # 重建损失权重
    lambda_adv: float = 1.0       # 对抗损失权重
    lambda_cycle: float = 5.0     # 循环一致性损失权重
    lambda_identity: float = 2.0  # 身份损失权重
    
    # 训练策略
    discriminator_steps: int = 1  # 每个生成器步骤对应的判别器步骤
    warmup_epochs: int = 5       # 预热轮数
    
    # 设备配置
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # 输出路径
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    result_dir: str = "results"
    
    # 日志和保存
    log_interval: int = 10       # 日志记录间隔
    save_interval: int = 10       # 模型保存间隔
    eval_interval: int = 5        # 评估间隔
    
    # 可视化
    vis_samples: int = 4          # 可视化样本数量
    
    # 数据增强
    use_rotation: bool = True     # 是否使用旋转增强
    use_jitter: bool = True       # 是否使用抖动增强
    use_scaling: bool = True      # 是否使用缩放增强
    rotation_range: float = 0.1   # 旋转范围
    jitter_std: float = 0.01      # 抖动标准差
    scaling_range: Tuple[float, float] = (0.9, 1.1)  # 缩放范围
    
    # 验证集划分
    val_split: float = 0.1        # 验证集比例
    test_split: float = 0.1       # 测试集比例
    
    def __post_init__(self):
        """创建必要的目录"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)


# 全局配置实例
config = Config()