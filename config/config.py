import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Config:
    """改进的配置类"""
    
    # 数据配置
    data_root: str = "datasets"
    sim_data_dir: str = "datasets/simulation"
    real_data_dir: str = "datasets/real_world"
    processed_data_dir: str = "datasets/processed"
    
    # 点云参数
    total_points: int = 120000    # 完整点云点数
    chunk_size: int = 2048        # 每个块的点数（减小以提高稳定性）
    overlap_ratio: float = 0.3    # 块之间的重叠率（增加重叠）
    num_chunks_per_pc: int = 80   # 每个点云的块数
    
    # 模型参数 - Diffusion Model
    model_type: str = "diffusion" 
    pointnet_channels: List[int] = (64, 128, 256, 512)
    latent_dim: int = 512
    time_embed_dim: int = 256
    num_timesteps: int = 1000      # Diffusion步数
    beta_schedule: str = "cosine"   # 噪声调度
    
    # 训练参数
    batch_size: int = 8            # 可以增大，因为Diffusion更稳定
    num_epochs: int = 40
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    ema_decay: float = 0.995       # EMA用于稳定训练
    gradient_clip: float = 1.0
    
    # 渐进式训练
    progressive_training: bool = True
    initial_chunks: int = 10       # 开始时只用10个块
    chunks_increment: int = 10     # 每个阶段增加10个块
    progressive_epochs: int = 20   # 每个阶段的训练轮数
    
    # 损失权重
    lambda_reconstruction: float = 1.0
    lambda_perceptual: float = 0.5
    lambda_continuity: float = 0.5
    lambda_boundary: float = 1.0   # 边界平滑损失
    
    # 设备配置
    device: str = "cuda"
    num_workers: int = 4
    
    # 输出路径
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    result_dir: str = "results"
    
    def __post_init__(self):
        """创建必要的目录"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

