"""
可视化工具
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from typing import List, Optional, Tuple


class PointCloudVisualizer:
    """点云可视化器"""
    
    def __init__(self):
        self.colors = {
            'sim': [0.1, 0.1, 0.8],      # 蓝色
            'real': [0.8, 0.1, 0.1],     # 红色
            'generated': [0.1, 0.8, 0.1]  # 绿色
        }
    
    def plot_style_transfer_result(self, sim_points: np.ndarray,
                                 generated_points: np.ndarray,
                                 real_points: np.ndarray,
                                 title: str = "Style Transfer Result",
                                 save_path: Optional[str] = None,
                                 sample_size: int = 5000):
        """
        绘制风格转换结果对比图
        """
        fig = plt.figure(figsize=(15, 5))
        
        # 采样点用于可视化
        if len(sim_points) > sample_size:
            idx = np.random.choice(len(sim_points), sample_size, replace=False)
            sim_sample = sim_points[idx]
            gen_sample = generated_points[idx]
        else:
            sim_sample = sim_points
            gen_sample = generated_points
        
        if len(real_points) > sample_size:
            idx = np.random.choice(len(real_points), sample_size, replace=False)
            real_sample = real_points[idx]
        else:
            real_sample = real_points
        
        # 仿真点云
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(sim_sample[:, 0], sim_sample[:, 1], sim_sample[:, 2],
                   c=self.colors['sim'], s=1, alpha=0.5)
        ax1.set_title('Simulation')
        self._set_axes_equal(ax1)
        
        # 生成的点云
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(gen_sample[:, 0], gen_sample[:, 1], gen_sample[:, 2],
                   c=self.colors['generated'], s=1, alpha=0.5)
        ax2.set_title('Generated (Real Style)')
        self._set_axes_equal(ax2)
        
        # 真实参考
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(real_sample[:, 0], real_sample[:, 1], real_sample[:, 2],
                   c=self.colors['real'], s=1, alpha=0.5)
        ax3.set_title('Real Reference')
        self._set_axes_equal(ax3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _set_axes_equal(self, ax):
        """设置3D坐标轴比例相等"""
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)
    
    def save_as_ply(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                    save_path: str = "pointcloud.ply"):
        """保存为PLY格式"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(save_path, pcd)
    
    def visualize_interactive(self, point_clouds: List[np.ndarray], 
                            labels: List[str], colors: Optional[List] = None):
        """交互式可视化多个点云"""
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        for i, (points, label) in enumerate(zip(point_clouds, labels)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            if colors and i < len(colors):
                pcd.paint_uniform_color(colors[i])
            else:
                pcd.paint_uniform_color([0.5, 0.5, 0.5])
            
            vis.add_geometry(pcd)
        
        vis.run()
        vis.destroy_window()
