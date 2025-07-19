import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
from typing import List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# 尝试导入seaborn，如果失败则设置标志
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None


class PointCloudVisualizer:
    """点云可视化工具"""
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器
        Args:
            style: matplotlib样式
            figsize: 图片大小
        """
        # 设置matplotlib样式，兼容不同版本
        try:
            if style == 'seaborn':
                # 尝试使用seaborn样式，如果失败则使用默认样式
                if HAS_SEABORN:
                    try:
                        sns.set_style("whitegrid")
                    except Exception:
                        pass
                plt.style.use('default')
            elif style in plt.style.available:
                plt.style.use(style)
            else:
                plt.style.use('default')
        except Exception:
            plt.style.use('default')
        
        self.figsize = figsize
        
        # 颜色配置
        self.colors = {
            'sim': '#FF6B6B',      # 红色 - 仿真数据
            'real': '#4ECDC4',     # 青色 - 真实数据  
            'generated': '#45B7D1', # 蓝色 - 生成数据
            'reference': '#96CEB4'  # 绿色 - 参考数据
        }
        
    def plot_point_cloud_3d(self, points: Union[np.ndarray, torch.Tensor],
                           colors: Optional[Union[np.ndarray, str]] = None,
                           title: str = "Point Cloud",
                           save_path: Optional[str] = None,
                           show_axes: bool = True,
                           point_size: int = 20) -> plt.Figure:
        """
        绘制3D点云
        Args:
            points: 点云数据 [N, 3]
            colors: 颜色数组或颜色名称
            title: 图片标题
            save_path: 保存路径
            show_axes: 是否显示坐标轴
            point_size: 点的大小
        Returns:
            matplotlib图形对象
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置颜色
        if colors is None:
            colors = self.colors['sim']
        elif isinstance(colors, str):
            colors = self.colors.get(colors, colors)
        
        # 绘制点云
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=colors, s=point_size, alpha=0.6)
        
        # 设置标题和标签
        ax.set_title(title, fontsize=14, fontweight='bold')
        if show_axes:
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_zlabel('Z', fontsize=12)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_comparison(self, point_clouds: List[Union[np.ndarray, torch.Tensor]],
                       labels: List[str],
                       title: str = "Point Cloud Comparison",
                       save_path: Optional[str] = None,
                       ncols: int = 2) -> plt.Figure:
        """
        对比显示多个点云
        Args:
            point_clouds: 点云列表
            labels: 标签列表
            title: 总标题
            save_path: 保存路径
            ncols: 列数
        Returns:
            matplotlib图形对象
        """
        n_clouds = len(point_clouds)
        nrows = (n_clouds + ncols - 1) // ncols
        
        fig = plt.figure(figsize=(self.figsize[0] * ncols, self.figsize[1] * nrows))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, (points, label) in enumerate(zip(point_clouds, labels)):
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()
            
            ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
            
            # 选择颜色
            color_key = 'sim' if 'sim' in label.lower() else 'real' if 'real' in label.lower() else 'generated'
            color = self.colors.get(color_key, self.colors['generated'])
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=color, s=10, alpha=0.6)
            
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_interactive_3d(self, points: Union[np.ndarray, torch.Tensor],
                           colors: Optional[Union[np.ndarray, str]] = None,
                           title: str = "Interactive Point Cloud",
                           save_path: Optional[str] = None) -> go.Figure:
        """
        创建交互式3D点云图
        Args:
            points: 点云数据 [N, 3]
            colors: 颜色数组或颜色名称
            title: 图片标题
            save_path: 保存路径（HTML）
        Returns:
            plotly图形对象
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        # 设置颜色
        if colors is None:
            colors = 'blue'
        elif isinstance(colors, str):
            colors = self.colors.get(colors, colors)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
                opacity=0.6
            ),
            name='Point Cloud'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_training_curves(self, metrics_history: dict,
                           title: str = "Training Curves",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制训练曲线
        Args:
            metrics_history: 指标历史字典
            title: 图片标题
            save_path: 保存路径
        Returns:
            matplotlib图形对象
        """
        # 确定子图数量
        n_metrics = len(metrics_history)
        ncols = 2
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, (metric_name, values) in enumerate(metrics_history.items()):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]
            
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, linewidth=2, label=metric_name)
            
            ax.set_title(metric_name.replace('_', ' ').title(), fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 隐藏未使用的子图
        for i in range(n_metrics, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_loss_landscape(self, loss_values: np.ndarray,
                           x_range: Tuple[float, float],
                           y_range: Tuple[float, float],
                           title: str = "Loss Landscape",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制损失景观
        Args:
            loss_values: 损失值矩阵 [H, W]
            x_range: X轴范围
            y_range: Y轴范围
            title: 图片标题
            save_path: 保存路径
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 创建网格
        x = np.linspace(x_range[0], x_range[1], loss_values.shape[1])
        y = np.linspace(y_range[0], y_range[1], loss_values.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # 绘制等高线图
        contour = ax.contour(X, Y, loss_values, levels=20, colors='black', alpha=0.4, linewidths=0.5)
        contourf = ax.contourf(X, Y, loss_values, levels=50, cmap='viridis', alpha=0.8)
        
        # 添加颜色条
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Loss Value', rotation=270, labelpad=20)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_metrics_comparison(self, metrics_dict: dict,
                              title: str = "Metrics Comparison",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制指标对比图
        Args:
            metrics_dict: 指标字典 {method_name: {metric_name: value}}
            title: 图片标题
            save_path: 保存路径
        Returns:
            matplotlib图形对象
        """
        # 转换为DataFrame
        df_data = []
        for method, metrics in metrics_dict.items():
            for metric, value in metrics.items():
                df_data.append({'Method': method, 'Metric': metric, 'Value': value})
        
        df = pd.DataFrame(df_data)
        
        # 获取唯一的指标
        unique_metrics = df['Metric'].unique()
        n_metrics = len(unique_metrics)
        
        # 创建子图
        ncols = 2
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(unique_metrics):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]
            
            metric_data = df[df['Metric'] == metric]
            
            # 绘制柱状图
            if HAS_SEABORN:
                try:
                    sns.barplot(data=metric_data, x='Method', y='Value', ax=ax)
                except Exception:
                    # 如果seaborn失败，使用matplotlib
                    ax.bar(metric_data['Method'], metric_data['Value'])
            else:
                ax.bar(metric_data['Method'], metric_data['Value'])
            ax.set_title(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
        
        # 隐藏未使用的子图
        for i in range(n_metrics, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_style_transfer_result(self, original: Union[np.ndarray, torch.Tensor],
                                 generated: Union[np.ndarray, torch.Tensor],
                                 reference: Union[np.ndarray, torch.Tensor],
                                 title: str = "Style Transfer Result",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制风格迁移结果
        Args:
            original: 原始点云
            generated: 生成点云
            reference: 参考点云
            title: 图片标题
            save_path: 保存路径
        Returns:
            matplotlib图形对象
        """
        point_clouds = [original, generated, reference]
        labels = ['Original (Simulation)', 'Generated (Real Style)', 'Reference (Real)']
        
        return self.plot_comparison(point_clouds, labels, title, save_path, ncols=3)
    
    def save_point_cloud(self, points: Union[np.ndarray, torch.Tensor],
                        save_path: str,
                        title: str = "Point Cloud",
                        color: str = 'sim',
                        **kwargs):
        """
        保存点云图片
        Args:
            points: 点云数据
            save_path: 保存路径
            title: 图片标题
            color: 颜色
            **kwargs: 其他参数
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 绘制并保存
        self.plot_point_cloud_3d(points, color, title, save_path, **kwargs)
    
    def create_animation_frames(self, point_cloud_sequence: List[Union[np.ndarray, torch.Tensor]],
                              save_dir: str,
                              title_prefix: str = "Frame") -> List[str]:
        """
        创建动画帧
        Args:
            point_cloud_sequence: 点云序列
            save_dir: 保存目录
            title_prefix: 标题前缀
        Returns:
            帧文件路径列表
        """
        os.makedirs(save_dir, exist_ok=True)
        frame_paths = []
        
        for i, points in enumerate(point_cloud_sequence):
            frame_path = os.path.join(save_dir, f"frame_{i:04d}.png")
            title = f"{title_prefix} {i+1}"
            
            self.save_point_cloud(points, frame_path, title)
            frame_paths.append(frame_path)
        
        return frame_paths


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.visualizer = PointCloudVisualizer()
        
    def plot_epoch_results(self, epoch: int,
                          sim_original: torch.Tensor,
                          sim_to_real: torch.Tensor,
                          real_reference: torch.Tensor,
                          metrics: dict):
        """
        绘制epoch结果
        Args:
            epoch: epoch编号
            sim_original: 原始仿真点云
            sim_to_real: 生成的真实风格点云
            real_reference: 真实参考点云
            metrics: 评估指标
        """
        # 创建epoch目录
        epoch_dir = os.path.join(self.log_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 绘制风格迁移结果
        for i in range(min(4, sim_original.size(0))):  # 最多显示4个样本
            result_path = os.path.join(epoch_dir, f"sample_{i}_result.png")
            self.visualizer.plot_style_transfer_result(
                sim_original[i].cpu().numpy(),
                sim_to_real[i].cpu().numpy(),
                real_reference[i].cpu().numpy(),
                title=f"Epoch {epoch} - Sample {i+1}",
                save_path=result_path
            )
        
        # 绘制指标图表
        if metrics:
            metrics_path = os.path.join(epoch_dir, "metrics.png")
            self.plot_metrics_summary(metrics, epoch, metrics_path)
    
    def plot_metrics_summary(self, metrics: dict, epoch: int, save_path: str):
        """
        绘制指标总结
        Args:
            metrics: 指标字典
            epoch: epoch编号
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Metrics Summary - Epoch {epoch}', fontsize=16)
        
        # 距离指标
        distance_metrics = {k: v for k, v in metrics.items() 
                          if 'distance' in k.lower() or 'chamfer' in k.lower()}
        if distance_metrics:
            ax = axes[0, 0]
            metrics_names = list(distance_metrics.keys())
            values = list(distance_metrics.values())
            ax.bar(metrics_names, values)
            ax.set_title('Distance Metrics')
            ax.tick_params(axis='x', rotation=45)
        
        # 其他指标...
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def test_visualization():
    """测试可视化功能"""
    # 创建测试数据
    n_points = 1000
    
    # 球形点云
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0.8, 1.2, n_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    sphere_points = np.column_stack([x, y, z])
    
    # 立方体点云
    cube_points = np.random.uniform(-1, 1, (n_points, 3))
    
    # 初始化可视化器
    visualizer = PointCloudVisualizer()
    
    # 测试单个点云绘制
    print("Testing single point cloud visualization...")
    visualizer.plot_point_cloud_3d(sphere_points, 'sim', "Test Sphere")
    
    # 测试对比绘制
    print("Testing comparison visualization...")
    visualizer.plot_comparison(
        [sphere_points, cube_points],
        ['Sphere', 'Cube'],
        "Point Cloud Comparison"
    )
    
    # 测试交互式绘制
    print("Testing interactive visualization...")
    visualizer.plot_interactive_3d(sphere_points, 'real', "Interactive Sphere")
    
    print("Visualization tests completed!")


if __name__ == "__main__":
    test_visualization()