"""
Training module for point cloud style transfer
"""

from .trainer import PointCloudStyleTransferTrainer

# 动态导入utils中的类，避免循环导入问题
def __getattr__(name):
    if name in ['AverageMeter', 'ProgressMeter', 'EarlyStopping', 'LearningRateScheduler', 
                'ModelCheckpoint', 'LossHistory', 'Timer', 'MetricsTracker', 'set_seed',
                'count_parameters', 'get_memory_usage', 'cleanup_memory']:
        from .utils import (
            AverageMeter, ProgressMeter, EarlyStopping, LearningRateScheduler,
            ModelCheckpoint, LossHistory, Timer, MetricsTracker, set_seed,
            count_parameters, get_memory_usage, cleanup_memory
        )
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'PointCloudStyleTransferTrainer',
    'AverageMeter',
    'ProgressMeter',
    'EarlyStopping', 
    'LearningRateScheduler',
    'ModelCheckpoint',
    'LossHistory',
    'Timer',
    'MetricsTracker',
    'set_seed',
    'count_parameters',
    'get_memory_usage',
    'cleanup_memory'
]