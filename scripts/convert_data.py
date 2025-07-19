#!/usr/bin/env python3
"""
数据格式转换工具
支持多种点云格式之间的转换
"""

import argparse
import os
import sys
import numpy as np
import glob
from tqdm import tqdm
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import load_point_cloud, save_point_cloud, normalize_point_cloud


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Point Cloud Data Format Converter')
    
    # 输入输出参数
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing point cloud files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for converted files')
    parser.add_argument('--input_format', type=str, required=True,
                       choices=['npy', 'npz', 'txt', 'ply', 'pcd', 'las'],
                       help='Input file format')
    parser.add_argument('--output_format', type=str, required=True,
                       choices=['npy', 'npz', 'txt', 'ply'],
                       help='Output file format')
    
    # 处理选项
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize point clouds')
    parser.add_argument('--normalization_method', type=str, default='unit_sphere',
                       choices=['unit_sphere', 'unit_cube', 'zero_mean'],
                       help='Normalization method')
    parser.add_argument('--remove_duplicates', action='store_true',
                       help='Remove duplicate points')
    parser.add_argument('--subsample', type=int, default=0,
                       help='Subsample to specified number of points (0 = no subsampling)')
    parser.add_argument('--filter_bounds', type=float, nargs=6, default=None,
                       metavar=('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'),
                       help='Filter points within specified bounds')
    
    # 输出选项
    parser.add_argument('--save_metadata', action='store_true',
                       help='Save conversion metadata')
    parser.add_argument('--preserve_structure', action='store_true',
                       help='Preserve directory structure in output')
    parser.add_argument('--max_files', type=int, default=0,
                       help='Maximum number of files to convert (0 = no limit)')
    
    return parser.parse_args()


def load_pcd_file(file_path: str) -> np.ndarray:
    """
    加载PCD格式文件
    Args:
        file_path: PCD文件路径
    Returns:
        点云数据
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        return points.astype(np.float32)
    except ImportError:
        raise ImportError("open3d is required for PCD file support. Install with: pip install open3d")


def load_las_file(file_path: str) -> np.ndarray:
    """
    加载LAS格式文件
    Args:
        file_path: LAS文件路径
    Returns:
        点云数据
    """
    try:
        import laspy
        las_file = laspy.read(file_path)
        points = np.column_stack([las_file.x, las_file.y, las_file.z])
        return points.astype(np.float32)
    except ImportError:
        raise ImportError("laspy is required for LAS file support. Install with: pip install laspy")


def load_ply_file(file_path: str) -> np.ndarray:
    """
    加载PLY格式文件
    Args:
        file_path: PLY文件路径
    Returns:
        点云数据
    """
    points = []
    with open(file_path, 'r') as f:
        # 读取头部信息
        in_header = True
        vertex_count = 0
        
        for line in f:
            if in_header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    in_header = False
                continue
            
            # 读取顶点数据
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except ValueError:
                    continue
            
            if len(points) >= vertex_count:
                break
    
    return np.array(points, dtype=np.float32)


def load_point_cloud_extended(file_path: str, format: str) -> np.ndarray:
    """
    扩展的点云加载函数
    Args:
        file_path: 文件路径
        format: 文件格式
    Returns:
        点云数据
    """
    if format == 'pcd':
        return load_pcd_file(file_path)
    elif format == 'las':
        return load_las_file(file_path)
    elif format == 'ply':
        return load_ply_file(file_path)
    else:
        return load_point_cloud(file_path)


def remove_duplicate_points(points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """
    移除重复点
    Args:
        points: 输入点云
        tolerance: 容差
    Returns:
        去重后的点云
    """
    # 使用numpy的unique函数
    _, unique_indices = np.unique(np.round(points / tolerance), 
                                 axis=0, return_index=True)
    return points[unique_indices]


def filter_points_by_bounds(points: np.ndarray, bounds: list) -> np.ndarray:
    """
    根据边界过滤点
    Args:
        points: 输入点云 [N, 3]
        bounds: 边界 [xmin, xmax, ymin, ymax, zmin, zmax]
    Returns:
        过滤后的点云
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    mask = ((points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
            (points[:, 2] >= zmin) & (points[:, 2] <= zmax))
    
    return points[mask]


def subsample_points(points: np.ndarray, target_size: int) -> np.ndarray:
    """
    点云下采样
    Args:
        points: 输入点云
        target_size: 目标点数
    Returns:
        下采样后的点云
    """
    if len(points) <= target_size:
        return points
    
    indices = np.random.choice(len(points), target_size, replace=False)
    return points[indices]


def process_single_file(file_path: str, output_path: str, args) -> dict:
    """
    处理单个文件
    Args:
        file_path: 输入文件路径
        output_path: 输出文件路径
        args: 参数对象
    Returns:
        处理统计信息
    """
    try:
        # 加载点云
        points = load_point_cloud_extended(file_path, args.input_format)
        original_size = len(points)
        
        # 处理流水线
        processing_stats = {
            'original_size': original_size,
            'final_size': 0,
            'operations': []
        }
        
        # 移除重复点
        if args.remove_duplicates:
            before_size = len(points)
            points = remove_duplicate_points(points)
            after_size = len(points)
            processing_stats['operations'].append({
                'operation': 'remove_duplicates',
                'before': before_size,
                'after': after_size,
                'removed': before_size - after_size
            })
        
        # 边界过滤
        if args.filter_bounds:
            before_size = len(points)
            points = filter_points_by_bounds(points, args.filter_bounds)
            after_size = len(points)
            processing_stats['operations'].append({
                'operation': 'filter_bounds',
                'before': before_size,
                'after': after_size,
                'removed': before_size - after_size
            })
        
        # 下采样
        if args.subsample > 0:
            before_size = len(points)
            points = subsample_points(points, args.subsample)
            after_size = len(points)
            processing_stats['operations'].append({
                'operation': 'subsample',
                'before': before_size,
                'after': after_size,
                'target_size': args.subsample
            })
        
        # 标准化
        normalization_params = None
        if args.normalize:
            points, normalization_params = normalize_point_cloud(
                points, args.normalization_method
            )
            processing_stats['operations'].append({
                'operation': 'normalize',
                'method': args.normalization_method,
                'params': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in normalization_params.items()}
            })
        
        processing_stats['final_size'] = len(points)
        
        # 保存处理后的点云
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_point_cloud(points, output_path, args.output_format)
        
        # 保存元数据（如果需要）
        if args.save_metadata:
            metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
            metadata = {
                'input_file': file_path,
                'output_file': output_path,
                'input_format': args.input_format,
                'output_format': args.output_format,
                'processing_stats': processing_stats,
                'normalization_params': normalization_params
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return processing_stats
        
    except Exception as e:
        raise Exception(f"Error processing {file_path}: {e}")


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取输入文件列表
    file_pattern = f"*.{args.input_format}"
    if args.preserve_structure:
        # 递归搜索
        input_files = []
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(f'.{args.input_format}'):
                    input_files.append(os.path.join(root, file))
    else:
        input_files = glob.glob(os.path.join(args.input_dir, file_pattern))
    
    if not input_files:
        print(f"No {args.input_format} files found in {args.input_dir}")
        return
    
    # 限制文件数量
    if args.max_files > 0:
        input_files = input_files[:args.max_files]
    
    print(f"Found {len(input_files)} files to convert")
    print(f"Converting from {args.input_format} to {args.output_format}")
    
    # 处理统计
    total_stats = {
        'processed': 0,
        'failed': 0,
        'total_original_points': 0,
        'total_final_points': 0
    }
    
    # 处理每个文件
    for file_path in tqdm(input_files, desc="Converting files"):
        try:
            # 确定输出路径
            if args.preserve_structure:
                # 保持目录结构
                rel_path = os.path.relpath(file_path, args.input_dir)
                output_path = os.path.join(args.output_dir, rel_path)
            else:
                # 平铺到输出目录
                filename = os.path.basename(file_path)
                output_path = os.path.join(args.output_dir, filename)
            
            # 修改文件扩展名
            output_path = os.path.splitext(output_path)[0] + f'.{args.output_format}'
            
            # 处理文件
            stats = process_single_file(file_path, output_path, args)
            
            # 更新统计
            total_stats['processed'] += 1
            total_stats['total_original_points'] += stats['original_size']
            total_stats['total_final_points'] += stats['final_size']
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            total_stats['failed'] += 1
            continue
    
    # 打印最终统计
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Total files processed: {total_stats['processed']}")
    print(f"Failed conversions: {total_stats['failed']}")
    print(f"Total original points: {total_stats['total_original_points']:,}")
    print(f"Total final points: {total_stats['total_final_points']:,}")
    
    if total_stats['total_original_points'] > 0:
        reduction_ratio = 1 - (total_stats['total_final_points'] / total_stats['total_original_points'])
        print(f"Point reduction ratio: {reduction_ratio:.2%}")
    
    print(f"Output directory: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()