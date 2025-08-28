from dataclasses import dataclass, field
import os
from typing import Tuple


@dataclass
class ConfigUnsupervised:
    """
    无监督模型的专属配置。
    通过修改此文件中的参数来控制两阶段的训练。
    """
    
    # --- 实验与路径 ---
    experiment_name: str = "test1"
    data_root: str = "datasets"
    processed_data_dir: str = os.path.join(data_root, "processed")
    
    # --- 数据与分块 ---
    total_points: int = 120000
    chunk_size: int = 4096
    overlap_ratio: float = 0.2
    
    # --- LiDAR特定配置 ---
    use_lidar_normalization: bool = True
    use_lidar_chunking: bool = True
    
    # --- 模型参数 ---
    time_embed_dim: int = 256
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    
    # --- 训练参数 ---
    num_epochs: int = 60 # 为每个阶段提供足够的训练轮数
    weight_decay: float = 1e-4
    ema_decay: float = 0.995
    gradient_clip: float = 1.0
    warmup_steps: int = 500 # 减少预热步骤
    
    # --- 自动批处理大小 ---
    _batch_size: int = None
    @property
    def batch_size(self) -> int:
        if self._batch_size is not None: return self._batch_size
        if self.chunk_size <= 2048: return 16
        if self.chunk_size <= 4096: return 8
        if self.chunk_size <= 8192: return 4
        return 1
    
    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value

    # --- 学习率 ---
    # 默认设置为第一阶段
    #learning_rate: float = 1e-4

    # # --- 损失权重 ---
    # # 默认设置为第一阶段
    # lambda_diffusion: float = 1.0
    # lambda_chamfer: float = 10.0
    # lambda_content: float = 1.0
    # lambda_style: float = 0.0  # 在第一阶段完全禁用风格损失
    # lambda_lidar_structure: float = 1.0
    # lambda_smooth: float = 0.5
    
    
    # 第二阶段
    learning_rate: float = 1e-5
    
    lambda_diffusion: float = 1.0
    lambda_chamfer: float = 5.0
    lambda_content: float = 1.0
    lambda_style: float = 0.05
    lambda_lidar_structure: float = 1.0
    lambda_smooth: float = 0.5
    
    # ADDED: 添加回缺失的数据增强参数
    augmentation_rotation_range: float = 0.05
    augmentation_jitter_std: float = 0.005
    augmentation_scale_range: Tuple[float, float] = (0.98, 1.02)
    
    # --- 运行配置 ---
    device: str = "cuda"
    num_workers: int = 4
    save_interval: int = 5
    log_interval: int = 50
    eval_interval: int = 1
    
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