__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 主要模块导入
from .config.config import Config
from .models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
from .training.trainer import DiffusionTrainer

__all__ = [
    "Config",
    "PointCloudDiffusionModel", 
    "DiffusionProcess",
    "DiffusionTrainer"
]
