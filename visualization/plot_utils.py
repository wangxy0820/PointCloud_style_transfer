import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from typing import List, Tuple, Optional, Union, Dict, Any
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

# 尝试导入seaborn，如果失败则设置标志
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None


def setup_matplotlib_style(style: str = 'default', dpi: int = 300):
    """
    设置matplotlib样式
    Args:
        style: 样式名称
        dpi: 图片分辨率
    """
    try:
        if style == 'seaborn':
            # 兼容处理seaborn样式
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
    
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def create_color_palette(name: str = 'default', n_colors: int = 10) -> List[str]:
    """
    创建颜色调色板
    Args:
        name: 调色板名称
        n_colors: 颜色数量
    Returns:
        颜色列表
    """
    if HAS_SEABORN and sns is not None:
        try:
            if name == 'default':
                return sns.color_palette("husl", n_colors)
            elif name == 'pastel':
                return sns.color_palette("pastel", n_colors)
            elif name == 'bright':
                return sns.color_palette("bright", n_colors)
            elif name == 'dark':
                return sns.color_palette("dark", n_colors)
            elif name == 'colorblind':
                return sns.color_palette("colorblind", n_colors)
            elif name == 'point_cloud':
                # 专门为点云设计的调色板
                return ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
            else:
                return sns.color_palette(name, n_colors)
        except Exception:
            pass
    
    # 如果没有seaborn或出错，使用基础颜色
    basic_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    if name == 'point_cloud':
        return basic_colors
    # 循环使用基础颜色
    return [basic_colors[i % len(basic_colors)] for i in range(n_colors)]


def create_custom_colormap(colors: List[str], name: str = 'custom') -> LinearSegmentedColormap:
    """
    创建自定义颜色映射
    Args:
        colors: 颜色列表
        name: 颜色映射名称
    Returns:
        颜色映射对象
    """
    return LinearSegmentedColormap.from_list(name, colors)


def plot_loss_curves(loss_history: Dict[str, List[float]], 
                     save_path: Optional[str] = None,
                     title: str = "Training Loss Curves",
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    绘制损失曲线
    Args:
        loss_history: 损失历史字典
        save_path: 保存路径
        title: 图片标题
        figsize: 图片大小
    Returns:
        matplotlib图形对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 生成器损失
    if 'generator' in loss_history:
        axes[0, 0].plot(loss_history['generator'], color='#FF6B6B', linewidth=2)
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 判别器损失
    if 'discriminator' in loss_history:
        axes[0, 1].plot(loss_history['discriminator'], color='#4ECDC4', linewidth=2)
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 循环一致性损失
    if 'cycle' in loss_history:
        axes[1, 0].plot(loss_history['cycle'], color='#45B7D1', linewidth=2)
        axes[1, 0].set_title('Cycle Consistency Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 总损失或其他指标
    if 'total' in loss_history:
        axes[1, 1].plot(loss_history['total'], color='#96CEB4', linewidth=2)
        axes[1, 1].set_title('Total Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    elif 'chamfer_distance' in loss_history:
        axes[1, 1].plot(loss_history['chamfer_distance'], color='#FFEAA7', linewidth=2)
        axes[1, 1].set_title('Chamfer Distance')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                          save_path: Optional[str] = None,
                          title: str = "Metrics Comparison",
                          figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    绘制指标对比图
    Args:
        metrics_dict: 指标字典 {method_name: {metric_name: value}}
        save_path: 保存路径
        title: 图片标题
        figsize: 图片大小
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
    
    # 计算子图布局
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    colors = create_color_palette('point_cloud', len(df['Method'].unique()))
    
    for i, metric in enumerate(unique_metrics):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        metric_data = df[df['Metric'] == metric]
        
        # 绘制柱状图
        bars = ax.bar(metric_data['Method'], metric_data['Value'], color=colors[:len(metric_data)])
        
        # 添加数值标签
        for bar, value in zip(bars, metric_data['Value']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏未使用的子图
    for i in range(n_metrics, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_progress(epochs: List[int], 
                         train_losses: List[float], 
                         val_losses: List[float],
                         save_path: Optional[str] = None,
                         title: str = "Training Progress",
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    绘制训练进度
    Args:
        epochs: epoch列表
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
        title: 图片标题
        figsize: 图片大小
    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, train_losses, label='Training Loss', color='#FF6B6B', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_losses, label='Validation Loss', color='#4ECDC4', linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # 添加最低点标注
    min_val_idx = np.argmin(val_losses)
    ax.annotate(f'Best: {val_losses[min_val_idx]:.4f}', 
                xy=(epochs[min_val_idx], val_losses[min_val_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         save_path: Optional[str] = None,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    绘制混淆矩阵
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图片标题
        figsize: 图片大小
    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 标准化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制热力图
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.set_title(title, fontweight='bold')
    
    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    # 设置标签
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    
    # 添加数值标注
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_distribution_comparison(data1: np.ndarray, data2: np.ndarray,
                               labels: List[str] = ['Data 1', 'Data 2'],
                               save_path: Optional[str] = None,
                               title: str = "Distribution Comparison",
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    绘制分布对比图
    Args:
        data1: 数据1
        data2: 数据2
        labels: 数据标签
        save_path: 保存路径
        title: 图片标题
        figsize: 图片大小
    Returns:
        matplotlib图形对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4']
    
    # 直方图对比
    axes[0, 0].hist(data1.flatten(), bins=50, alpha=0.7, color=colors[0], label=labels[0], density=True)
    axes[0, 0].hist(data2.flatten(), bins=50, alpha=0.7, color=colors[1], label=labels[1], density=True)
    axes[0, 0].set_title('Histogram Comparison')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 箱线图对比
    box_data = [data1.flatten(), data2.flatten()]
    bp = axes[0, 1].boxplot(box_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 1].set_title('Box Plot Comparison')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 密度图对比
    axes[1, 0].hist(data1.flatten(), bins=50, alpha=0.7, color=colors[0], label=labels[0], density=True, histtype='step', linewidth=2)
    axes[1, 0].hist(data2.flatten(), bins=50, alpha=0.7, color=colors[1], label=labels[1], density=True, histtype='step', linewidth=2)
    axes[1, 0].set_title('Density Comparison')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 累积分布函数
    sorted_data1 = np.sort(data1.flatten())
    sorted_data2 = np.sort(data2.flatten())
    p1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
    p2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)
    
    axes[1, 1].plot(sorted_data1, p1, color=colors[0], label=labels[0], linewidth=2)
    axes[1, 1].plot(sorted_data2, p2, color=colors[1], label=labels[1], linewidth=2)
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_heatmap(data: np.ndarray,
                x_labels: Optional[List[str]] = None,
                y_labels: Optional[List[str]] = None,
                save_path: Optional[str] = None,
                title: str = "Heatmap",
                figsize: Tuple[int, int] = (10, 8),
                cmap: str = 'viridis') -> plt.Figure:
    """
    绘制热力图
    Args:
        data: 数据矩阵
        x_labels: x轴标签
        y_labels: y轴标签
        save_path: 保存路径
        title: 图片标题
        figsize: 图片大小
        cmap: 颜色映射
    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # 设置标签
    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    ax.set_title(title, fontweight='bold')
    
    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    # 添加数值标注（如果数据不太大）
    if data.shape[0] <= 20 and data.shape[1] <= 20:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_scatter_with_regression(x: np.ndarray, y: np.ndarray,
                                save_path: Optional[str] = None,
                                title: str = "Scatter Plot with Regression",
                                xlabel: str = "X",
                                ylabel: str = "Y",
                                figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    绘制带回归线的散点图
    Args:
        x: x轴数据
        y: y轴数据
        save_path: 保存路径
        title: 图片标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图片大小
    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 散点图
    ax.scatter(x, y, alpha=0.6, color='#45B7D1', s=30)
    
    # 回归线
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
    
    # 计算相关系数
    correlation = np.corrcoef(x, y)[0, 1]
    
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(f'{title}\nCorrelation: {correlation:.3f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_subplot_grid(n_plots: int, ncols: int = 3) -> Tuple[plt.Figure, np.ndarray]:
    """
    创建子图网格
    Args:
        n_plots: 子图数量
        ncols: 列数
    Returns:
        图形对象和坐标轴数组
    """
    nrows = (n_plots + ncols - 1) // ncols
    figsize = (ncols * 4, nrows * 3)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # 确保axes是2D数组
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # 隐藏多余的子图
    for i in range(n_plots, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    return fig, axes


def save_figure_with_metadata(fig: plt.Figure, save_path: str, metadata: Dict[str, Any]):
    """
    保存图片并添加元数据
    Args:
        fig: matplotlib图形对象
        save_path: 保存路径
        metadata: 元数据字典
    """
    # 添加元数据到图片
    fig.savefig(save_path, dpi=300, bbox_inches='tight', metadata=metadata)
    
    # 保存元数据到JSON文件
    metadata_path = save_path.replace('.png', '_metadata.json').replace('.jpg', '_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def create_animation_frames(data_sequence: List[np.ndarray],
                          plot_func: callable,
                          save_dir: str,
                          prefix: str = 'frame') -> List[str]:
    """
    创建动画帧
    Args:
        data_sequence: 数据序列
        plot_func: 绘图函数
        save_dir: 保存目录
        prefix: 文件名前缀
    Returns:
        帧文件路径列表
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    frame_paths = []
    for i, data in enumerate(data_sequence):
        frame_path = os.path.join(save_dir, f'{prefix}_{i:04d}.png')
        fig = plot_func(data, title=f'Frame {i+1}')
        fig.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)
    
    return frame_paths


# 常用颜色和样式常量
COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4', 
    'accent': '#45B7D1',
    'success': '#96CEB4',
    'warning': '#FFEAA7',
    'error': '#E74C3C',
    'info': '#74B9FF',
    'dark': '#2D3436',
    'light': '#DDD'
}

STYLE_CONFIGS = {
    'default': {
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    },
    'presentation': {
        'figure.figsize': (12, 8),
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    },
    'paper': {
        'figure.figsize': (8, 6),
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    }
}


def apply_style_config(style_name: str = 'default'):
    """
    应用样式配置
    Args:
        style_name: 样式名称
    """
    if style_name in STYLE_CONFIGS:
        plt.rcParams.update(STYLE_CONFIGS[style_name])
    else:
        print(f"Unknown style: {style_name}. Available styles: {list(STYLE_CONFIGS.keys())}")