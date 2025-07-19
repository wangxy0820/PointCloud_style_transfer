"""
Point Cloud Style Transfer Project

A comprehensive framework for point cloud style transfer using PointNet++ and GAN.
Supports large-scale point clouds (120k points) with chunking strategies and 
cycle-consistent domain adaptation from simulation to real world.
"""

__version__ = "1.0.0"
__author__ = "Point Cloud Style Transfer Team"
__email__ = "contact@example.com"

# 简化导入，只导入最常用的组件
from config import Config

__all__ = [
    'Config'
]