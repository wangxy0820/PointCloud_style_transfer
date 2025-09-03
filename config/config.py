# config/config.py

from dataclasses import dataclass
import os
import torch

@dataclass
class Config:
    
    # 实验设置
    experiment_name: str = "hierarchical_test"
    data_root: str = "datasets"
    processed_data_dir: str = os.path.join(data_root, "processed_hierarchical")
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"
    
    # 分层数据参数 
    total_points: int = 120000  # 完整点云大小
    global_points: int = 30000  # 全局下采样大小

    
    # 模型参数
    time_embed_dim: int = 256
    feature_dim: int = 512
    global_feature_dim: int = 256
    
    # Diffusion参数
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    noise_schedule_offset: float = 0.0008
    
    # 训练参数
    num_epochs: int = 100
    learning_rate: float = 5e-4  
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    gradient_clip: float = 1.0
    
    # 学习率调度
    lr_scheduler: str = "cosine_with_warmup"
    warmup_epochs: int = 20
    min_lr_ratio: float = 0.01
    
    # 批处理
    batch_size: int = 8 
    num_workers: int = 2
    use_amp: bool = True
    
    # 验证和保存
    val_interval: int = 5
    save_interval: int = 10
    
    # 损失函数
    loss_scale_factor: float = 1.0
    use_hierarchical: bool = True 
    
    def __post_init__(self):
        for dir_path in [self.log_dir, self.result_dir, self.processed_data_dir]:
            os.makedirs(dir_path, exist_ok=True)