import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Config:
    """改进的配置类 - 支持LiDAR点云处理"""
    
    # 数据配置
    data_root: str = "datasets"
    sim_data_dir: str = "datasets/simulation"
    real_data_dir: str = "datasets/real_world"
    processed_data_dir: str = "datasets/processed"
    experiment_dir: str = "test1"
    
    # 点云参数 - 修改以支持更大的chunk
    total_points: int = 120000    # 完整点云点数
    chunk_size: int = 4096        # 增大到4096（可选：2048, 4096, 8192, 16384）
    overlap_ratio: float = 0.2    # 减小重叠率以适应更大的chunk
    num_chunks_per_pc: int = 40   # 减少块数因为每块更大了
    
    # LiDAR特定配置（新增）
    use_lidar_normalization: bool = True  # 使用LiDAR友好的标准化
    use_lidar_chunking: bool = True       # 使用LiDAR感知的分块策略
    
    # 根据chunk_size自动调整batch_size
    @property
    def auto_batch_size(self) -> int:
        """根据chunk_size自动计算合适的batch_size"""
        if self.chunk_size <= 2048:
            return 8
        elif self.chunk_size <= 4096:
            return 4
        elif self.chunk_size <= 8192:
            return 2
        else:  # 16384 or larger
            return 1
    
    # 模型参数 - Diffusion Model
    model_type: str = "diffusion" 
    pointnet_channels: List[int] = (64, 128, 256, 512)
    latent_dim: int = 512
    time_embed_dim: int = 256
    num_timesteps: int = 100      # Diffusion步数
    beta_schedule: str = "linear"   # 噪声调度
    
    # 训练参数 - 使用自动batch_size
    @property
    def batch_size(self) -> int:
        return self._batch_size if hasattr(self, '_batch_size') else self.auto_batch_size
    
    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value
    
    # 默认训练参数
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
    
    # 损失权重 - 针对LiDAR调整
    lambda_reconstruction: float = 1.0
    lambda_perceptual: float = 0.5
    lambda_continuity: float = 0.5
    lambda_boundary: float = 1.0
    lambda_content: float = 2.0         # 减小内容权重
    lambda_style: float = 0.01          # 进一步减小风格权重
    lambda_lidar_structure: float = 0.5 # 大幅减小LiDAR结构损失权重
    
    # 数据增强参数 - 针对LiDAR调整
    augmentation_rotation_range: float = 0.05   # 减小旋转范围
    augmentation_jitter_std: float = 0.005      # 减小抖动
    augmentation_scale_range: Tuple[float, float] = (0.98, 1.02)  # 减小缩放范围
    
    # 设备配置
    device: str = "cuda"
    num_workers: int = 4
    
    # 训练配置
    save_interval: int = 10        # 保存检查点间隔
    log_interval: int = 50         # 日志记录间隔
    eval_interval: int = 1         # 验证间隔
    
    # 输出路径
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    result_dir: str = "results"
    
    def __post_init__(self):
        """创建必要的目录并验证配置"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # 验证chunk配置
        self._validate_chunk_config()
        
        # 打印配置信息
        print(f"Configuration initialized:")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  LiDAR mode: {self.use_lidar_normalization}")
        print(f"  Estimated memory per batch: ~{self._estimate_memory_usage():.1f} GB")
    
    def _validate_chunk_config(self):
        """验证chunk配置的合理性"""
        estimated_chunks = self.total_points // (self.chunk_size * (1 - self.overlap_ratio))
        if estimated_chunks < 10:
            print(f"Warning: Only ~{int(estimated_chunks)} chunks will be created. Consider reducing chunk_size.")
        if estimated_chunks > 200:
            print(f"Warning: ~{int(estimated_chunks)} chunks will be created. Consider increasing chunk_size.")
    
    def _estimate_memory_usage(self) -> float:
        """估算GPU内存使用（GB）"""
        # 粗略估算：每个点3个float32，考虑模型和梯度
        points_per_batch = self.batch_size * self.chunk_size * 3 * 4  # bytes
        model_overhead = 2.0  # GB for model weights and gradients
        return (points_per_batch / 1e9) * 10 + model_overhead  # x10 for intermediate tensors