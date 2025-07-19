import numpy as np
import torch
import os
import glob
from typing import List, Tuple, Union, Optional
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_point_cloud(file_path: str) -> np.ndarray:
    """
    加载点云文件
    Args:
        file_path: 文件路径
    Returns:
        点云数据 [N, 3]
    """
    if file_path.endswith('.npy'):
        points = np.load(file_path)
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        points = data['points'] if 'points' in data else data[data.files[0]]
    elif file_path.endswith('.txt'):
        points = np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # 确保是浮点型且只保留xyz坐标
    points = points.astype(np.float32)
    if points.shape[1] > 3:
        points = points[:, :3]
    
    return points


def save_point_cloud(points: np.ndarray, file_path: str, format: str = 'npy'):
    """
    保存点云文件
    Args:
        points: 点云数据 [N, 3]
        file_path: 保存路径
        format: 文件格式
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if format == 'npy':
        np.save(file_path, points)
    elif format == 'npz':
        np.savez_compressed(file_path, points=points)
    elif format == 'txt':
        np.savetxt(file_path, points, fmt='%.6f')
    elif format == 'ply':
        save_ply(points, file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_ply(points: np.ndarray, file_path: str, colors: Optional[np.ndarray] = None):
    """
    保存PLY格式点云文件
    Args:
        points: 点云数据 [N, 3]
        file_path: 文件路径
        colors: 颜色数据 [N, 3] (可选)
    """
    with open(file_path, 'w') as f:
        # PLY头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # 点云数据
        for i, point in enumerate(points):
            if colors is not None:
                color = colors[i]
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                       f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def normalize_point_cloud(points: np.ndarray, method: str = 'unit_sphere') -> Tuple[np.ndarray, dict]:
    """
    标准化点云
    Args:
        points: 输入点云 [N, 3]
        method: 标准化方法 ('unit_sphere', 'unit_cube', 'zero_mean')
    Returns:
        标准化后的点云和标准化参数
    """
    # 计算中心
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    normalization_params = {'centroid': centroid}
    
    if method == 'unit_sphere':
        # 缩放到单位球
        max_dist = np.max(np.linalg.norm(centered_points, axis=1))
        if max_dist > 0:
            scale = 1.0 / max_dist
            normalized_points = centered_points * scale
        else:
            scale = 1.0
            normalized_points = centered_points
        normalization_params['scale'] = scale
        
    elif method == 'unit_cube':
        # 缩放到单位立方体
        max_range = np.max(np.abs(centered_points))
        if max_range > 0:
            scale = 1.0 / max_range
            normalized_points = centered_points * scale
        else:
            scale = 1.0
            normalized_points = centered_points
        normalization_params['scale'] = scale
        
    elif method == 'zero_mean':
        # 只中心化，不缩放
        normalized_points = centered_points
        normalization_params['scale'] = 1.0
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_points, normalization_params


def denormalize_point_cloud(points: np.ndarray, normalization_params: dict) -> np.ndarray:
    """
    反标准化点云
    Args:
        points: 标准化的点云 [N, 3]
        normalization_params: 标准化参数
    Returns:
        原始尺度的点云
    """
    scale = normalization_params.get('scale', 1.0)
    centroid = normalization_params.get('centroid', np.zeros(3))
    
    # 反缩放和反中心化
    denormalized_points = points / scale + centroid
    
    return denormalized_points


def random_rotation_matrix(max_angle: float = np.pi) -> np.ndarray:
    """
    生成随机旋转矩阵
    Args:
        max_angle: 最大旋转角度
    Returns:
        3x3旋转矩阵
    """
    # 随机旋转轴
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    
    # 随机旋转角度
    angle = np.random.uniform(-max_angle, max_angle)
    
    # 使用Rodrigues公式计算旋转矩阵
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # 反对称矩阵
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    # 旋转矩阵
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
    
    return R


def augment_point_cloud(points: np.ndarray,
                       rotation_range: float = 0.1,
                       noise_std: float = 0.01,
                       scale_range: Tuple[float, float] = (0.9, 1.1),
                       dropout_ratio: float = 0.0) -> np.ndarray:
    """
    点云数据增强
    Args:
        points: 输入点云 [N, 3]
        rotation_range: 旋转范围（弧度）
        noise_std: 噪声标准差
        scale_range: 缩放范围
        dropout_ratio: 随机丢弃点的比例
    Returns:
        增强后的点云
    """
    augmented_points = points.copy()
    
    # 随机旋转
    if rotation_range > 0:
        R = random_rotation_matrix(rotation_range)
        augmented_points = augmented_points @ R.T
    
    # 添加噪声
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, augmented_points.shape)
        augmented_points += noise
    
    # 随机缩放
    if scale_range[0] != 1.0 or scale_range[1] != 1.0:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        augmented_points *= scale
    
    # 随机丢弃点
    if dropout_ratio > 0:
        num_points = len(augmented_points)
        num_keep = int(num_points * (1 - dropout_ratio))
        keep_indices = np.random.choice(num_points, num_keep, replace=False)
        augmented_points = augmented_points[keep_indices]
    
    return augmented_points


def compute_point_cloud_bounds(points: np.ndarray) -> dict:
    """
    计算点云边界信息
    Args:
        points: 点云 [N, 3]
    Returns:
        边界信息字典
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    
    return {
        'min': min_coords,
        'max': max_coords,
        'center': center,
        'size': size,
        'volume': np.prod(size),
        'diagonal': np.linalg.norm(size)
    }


def remove_outliers(points: np.ndarray, k: int = 20, std_threshold: float = 2.0) -> np.ndarray:
    """
    移除点云中的离群点
    Args:
        points: 输入点云 [N, 3]
        k: K近邻数量
        std_threshold: 标准差阈值
    Returns:
        去除离群点后的点云
    """
    if len(points) <= k:
        return points
    
    # 计算每个点的K近邻距离
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)  # +1因为包含自身
    distances, _ = nbrs.kneighbors(points)
    
    # 计算到K近邻的平均距离（排除自身）
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # 计算全局统计
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # 移除离群点
    threshold = global_mean + std_threshold * global_std
    inlier_mask = mean_distances <= threshold
    
    return points[inlier_mask]


def subsample_point_cloud(points: np.ndarray, target_size: int, method: str = 'random') -> np.ndarray:
    """
    点云下采样
    Args:
        points: 输入点云 [N, 3]
        target_size: 目标点数
        method: 采样方法 ('random', 'farthest', 'uniform')
    Returns:
        下采样后的点云
    """
    if len(points) <= target_size:
        return points
    
    if method == 'random':
        indices = np.random.choice(len(points), target_size, replace=False)
        return points[indices]
    
    elif method == 'farthest':
        return farthest_point_sampling(points, target_size)
    
    elif method == 'uniform':
        step = len(points) // target_size
        indices = np.arange(0, len(points), step)[:target_size]
        return points[indices]
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    最远点采样
    Args:
        points: 输入点云 [N, 3]
        num_samples: 采样点数
    Returns:
        采样后的点云
    """
    if num_samples >= len(points):
        return points
    
    # 初始化
    sampled_indices = [0]  # 从第一个点开始
    remaining_indices = set(range(1, len(points)))
    
    for _ in range(num_samples - 1):
        # 计算剩余点到已采样点的最小距离
        max_min_distance = -1
        best_index = -1
        
        for idx in remaining_indices:
            min_distance = float('inf')
            for sampled_idx in sampled_indices:
                distance = np.linalg.norm(points[idx] - points[sampled_idx])
                min_distance = min(min_distance, distance)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_index = idx
        
        sampled_indices.append(best_index)
        remaining_indices.remove(best_index)
    
    return points[sampled_indices]


def merge_point_clouds(point_clouds: List[np.ndarray], method: str = 'concatenate') -> np.ndarray:
    """
    合并多个点云
    Args:
        point_clouds: 点云列表
        method: 合并方法 ('concatenate', 'interleave', 'weighted')
    Returns:
        合并后的点云
    """
    if not point_clouds:
        return np.array([]).reshape(0, 3)
    
    if len(point_clouds) == 1:
        return point_clouds[0]
    
    if method == 'concatenate':
        return np.concatenate(point_clouds, axis=0)
    
    elif method == 'interleave':
        # 交错合并
        max_len = max(len(pc) for pc in point_clouds)
        merged = []
        
        for i in range(max_len):
            for pc in point_clouds:
                if i < len(pc):
                    merged.append(pc[i])
        
        return np.array(merged)
    
    elif method == 'weighted':
        # 加权平均（要求所有点云大小相同）
        if not all(len(pc) == len(point_clouds[0]) for pc in point_clouds):
            raise ValueError("All point clouds must have the same size for weighted merging")
        
        weights = np.ones(len(point_clouds)) / len(point_clouds)
        merged = np.zeros_like(point_clouds[0])
        
        for i, pc in enumerate(point_clouds):
            merged += weights[i] * pc
        
        return merged
    
    else:
        raise ValueError(f"Unknown merging method: {method}")


def compute_point_cloud_statistics(points: np.ndarray) -> dict:
    """
    计算点云统计信息
    Args:
        points: 点云 [N, 3]
    Returns:
        统计信息字典
    """
    stats = {}
    
    # 基本统计
    stats['num_points'] = len(points)
    stats['mean'] = np.mean(points, axis=0)
    stats['std'] = np.std(points, axis=0)
    stats['min'] = np.min(points, axis=0)
    stats['max'] = np.max(points, axis=0)
    
    # 几何统计
    centroid = stats['mean']
    distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
    stats['mean_distance_to_centroid'] = np.mean(distances_to_centroid)
    stats['max_distance_to_centroid'] = np.max(distances_to_centroid)
    
    # 密度统计（使用K近邻）
    if len(points) > 10:
        k = min(10, len(points) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        
        # 排除自身距离
        knn_distances = distances[:, 1:]
        stats['mean_knn_distance'] = np.mean(knn_distances)
        stats['std_knn_distance'] = np.std(knn_distances)
        stats['density_estimate'] = 1.0 / (np.mean(knn_distances) + 1e-8)
    
    return stats


def convert_to_torch(points: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    将numpy数组转换为PyTorch张量
    Args:
        points: numpy点云数组
        device: 目标设备
    Returns:
        PyTorch张量
    """
    return torch.from_numpy(points.astype(np.float32)).to(device)


def convert_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    将PyTorch张量转换为numpy数组
    Args:
        tensor: PyTorch张量
    Returns:
        numpy数组
    """
    return tensor.detach().cpu().numpy()


def batch_process_files(input_dir: str, output_dir: str, 
                       process_func, file_pattern: str = "*.npy",
                       batch_size: int = 32) -> None:
    """
    批量处理文件
    Args:
        input_dir: 输入目录
        output_dir: 输出目录  
        process_func: 处理函数
        file_pattern: 文件模式
        batch_size: 批次大小
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, file_pattern))
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}")
        
        for file_path in batch_files:
            try:
                result = process_func(file_path)
                
                # 保存结果
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_processed.npy")
                np.save(output_path, result)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue


def visualize_point_cloud_simple(points: np.ndarray, title: str = "Point Cloud", 
                                save_path: Optional[str] = None) -> None:
    """
    简单的点云可视化
    Args:
        points: 点云数据 [N, 3]
        title: 图片标题
        save_path: 保存路径（可选）
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_dataset_summary(data_dir: str, output_file: str = "dataset_summary.json") -> dict:
    """
    创建数据集摘要
    Args:
        data_dir: 数据集目录
        output_file: 输出文件名
    Returns:
        数据集摘要字典
    """
    import json
    
    summary = {
        'total_files': 0,
        'domains': {},
        'statistics': {}
    }
    
    # 遍历子目录
    for domain in os.listdir(data_dir):
        domain_path = os.path.join(data_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        
        files = glob.glob(os.path.join(domain_path, "*.npy"))
        summary['domains'][domain] = {
            'num_files': len(files),
            'files': [os.path.basename(f) for f in files[:10]]  # 只保存前10个文件名
        }
        
        # 计算统计信息
        if files:
            sample_file = files[0]
            points = load_point_cloud(sample_file)
            stats = compute_point_cloud_statistics(points)
            summary['domains'][domain]['sample_stats'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in stats.items()
            }
        
        summary['total_files'] += len(files)
    
    # 保存摘要
    summary_path = os.path.join(data_dir, output_file)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Dataset summary saved to: {summary_path}")
    return summary