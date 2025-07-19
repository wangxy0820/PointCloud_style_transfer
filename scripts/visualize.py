#!/usr/bin/env python3
"""
点云可视化脚本
用于可视化训练结果、数据集和推理结果
"""

import argparse
import os
import sys
import numpy as np
import glob
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.visualize import PointCloudVisualizer, TrainingVisualizer
import matplotlib.pyplot as plt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Point Cloud Visualization Tool')
    
    # 输入参数
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing point clouds or results')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualization images')
    
    # 可视化模式
    parser.add_argument('--mode', type=str, default='point_clouds',
                       choices=['point_clouds', 'comparison', 'training_curves', 
                               'style_transfer', 'dataset_overview', 'metrics'],
                       help='Visualization mode')
    
    # 点云文件参数
    parser.add_argument('--file_pattern', type=str, default='*.npy',
                       help='File pattern to match point cloud files')
    parser.add_argument('--max_files', type=int, default=20,
                       help='Maximum number of files to visualize')
    
    # 对比可视化参数
    parser.add_argument('--original_dir', type=str, default='',
                       help='Directory containing original point clouds (for comparison)')
    parser.add_argument('--generated_dir', type=str, default='',
                       help='Directory containing generated point clouds')
    parser.add_argument('--reference_dir', type=str, default='',
                       help='Directory containing reference point clouds')
    
    # 训练曲线参数
    parser.add_argument('--metrics_file', type=str, default='',
                       help='JSON file containing training metrics')
    parser.add_argument('--log_dir', type=str, default='',
                       help='TensorBoard log directory')
    
    # 可视化选项
    parser.add_argument('--interactive', action='store_true',
                       help='Create interactive 3D visualizations')
    parser.add_argument('--save_html', action='store_true',
                       help='Save interactive visualizations as HTML')
    parser.add_argument('--point_size', type=int, default=20,
                       help='Point size for visualization')
    parser.add_argument('--figure_size', type=int, nargs=2, default=[12, 8],
                       help='Figure size (width, height)')
    
    # 样式选项
    parser.add_argument('--style', type=str, default='seaborn',
                       help='Matplotlib style')
    parser.add_argument('--color_scheme', type=str, default='default',
                       choices=['default', 'pastel', 'bright', 'dark'],
                       help='Color scheme for visualizations')
    
    return parser.parse_args()


def visualize_point_clouds(args, visualizer):
    """可视化单个点云文件"""
    print("Visualizing individual point clouds...")
    
    # 获取文件列表
    pattern = os.path.join(args.input_dir, args.file_pattern)
    files = glob.glob(pattern)[:args.max_files]
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(files)} files to visualize")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'individual_point_clouds')
    os.makedirs(output_dir, exist_ok=True)
    
    for i, file_path in enumerate(files):
        try:
            # 加载点云
            points = np.load(file_path)
            if points.shape[1] > 3:
                points = points[:, :3]
            
            # 生成文件名
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # 静态可视化
            img_path = os.path.join(output_dir, f"{base_name}.png")
            visualizer.save_point_cloud(
                points, img_path,
                title=f"Point Cloud - {base_name}",
                point_size=args.point_size
            )
            
            # 交互式可视化（如果需要）
            if args.interactive:
                html_path = os.path.join(output_dir, f"{base_name}.html")
                fig = visualizer.plot_interactive_3d(
                    points, title=f"Interactive - {base_name}"
                )
                if args.save_html:
                    fig.write_html(html_path)
            
            print(f"Processed: {base_name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Individual visualizations saved to: {output_dir}")


def visualize_comparison(args, visualizer):
    """可视化对比结果"""
    print("Creating comparison visualizations...")
    
    if not all([args.original_dir, args.generated_dir]):
        print("Error: Both --original_dir and --generated_dir are required for comparison mode")
        return
    
    # 获取文件列表
    original_files = glob.glob(os.path.join(args.original_dir, "*.npy"))
    generated_files = glob.glob(os.path.join(args.generated_dir, "*.npy"))
    
    # 匹配文件名
    matched_pairs = []
    for orig_file in original_files:
        base_name = os.path.splitext(os.path.basename(orig_file))[0]
        # 寻找对应的生成文件
        for gen_file in generated_files:
            if base_name in os.path.basename(gen_file):
                matched_pairs.append((orig_file, gen_file))
                break
    
    if not matched_pairs:
        print("No matching file pairs found")
        return
    
    print(f"Found {len(matched_pairs)} matching pairs")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(output_dir, exist_ok=True)
    
    # 限制数量
    matched_pairs = matched_pairs[:args.max_files]
    
    for i, (orig_file, gen_file) in enumerate(matched_pairs):
        try:
            # 加载点云
            orig_points = np.load(orig_file)[:, :3]
            gen_points = np.load(gen_file)[:, :3]
            
            base_name = os.path.splitext(os.path.basename(orig_file))[0]
            
            # 加载参考点云（如果提供）
            ref_points = None
            if args.reference_dir:
                ref_files = glob.glob(os.path.join(args.reference_dir, f"*{base_name}*.npy"))
                if ref_files:
                    ref_points = np.load(ref_files[0])[:, :3]
            
            # 创建对比可视化
            if ref_points is not None:
                # 三方对比：原始、生成、参考
                comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
                visualizer.plot_style_transfer_result(
                    orig_points, gen_points, ref_points,
                    title=f"Style Transfer Result - {base_name}",
                    save_path=comparison_path
                )
            else:
                # 二方对比：原始 vs 生成
                comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
                visualizer.plot_comparison(
                    [orig_points, gen_points],
                    ['Original', 'Generated'],
                    title=f"Before vs After - {base_name}",
                    save_path=comparison_path
                )
            
            print(f"Processed comparison: {base_name}")
            
        except Exception as e:
            print(f"Error creating comparison for {orig_file}: {e}")
            continue
    
    print(f"Comparison visualizations saved to: {output_dir}")


def visualize_training_curves(args, visualizer):
    """可视化训练曲线"""
    print("Creating training curve visualizations...")
    
    metrics_data = {}
    
    # 从JSON文件加载指标
    if args.metrics_file and os.path.exists(args.metrics_file):
        with open(args.metrics_file, 'r') as f:
            metrics_data = json.load(f)
    
    # 从TensorBoard日志加载指标
    elif args.log_dir and os.path.exists(args.log_dir):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            # 获取所有事件文件
            event_files = glob.glob(os.path.join(args.log_dir, "events.out.tfevents.*"))
            
            for event_file in event_files:
                ea = EventAccumulator(event_file)
                ea.Reload()
                
                # 提取标量数据
                for tag in ea.Tags()['scalars']:
                    scalar_events = ea.Scalars(tag)
                    values = [event.value for event in scalar_events]
                    metrics_data[tag] = values
                    
        except ImportError:
            print("TensorBoard not available for log parsing")
            return
        except Exception as e:
            print(f"Error parsing TensorBoard logs: {e}")
            return
    
    else:
        print("No metrics file or log directory provided")
        return
    
    if not metrics_data:
        print("No metrics data found")
        return
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'training_curves')
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制训练曲线
    curves_path = os.path.join(output_dir, 'training_curves.png')
    visualizer.plot_training_curves(
        metrics_data,
        title="Training Progress",
        save_path=curves_path
    )
    
    print(f"Training curves saved to: {curves_path}")


def visualize_dataset_overview(args, visualizer):
    """可视化数据集概览"""
    print("Creating dataset overview...")
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(args.input_dir) 
              if os.path.isdir(os.path.join(args.input_dir, d))]
    
    if not subdirs:
        print("No subdirectories found in input directory")
        return
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'dataset_overview')
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个子目录创建可视化
    for subdir in subdirs:
        subdir_path = os.path.join(args.input_dir, subdir)
        files = glob.glob(os.path.join(subdir_path, "*.npy"))
        
        if not files:
            continue
        
        print(f"Processing {subdir} with {len(files)} files")
        
        # 随机选择几个样本
        sample_files = np.random.choice(files, min(6, len(files)), replace=False)
        
        sample_points = []
        sample_labels = []
        
        for file_path in sample_files:
            try:
                points = np.load(file_path)[:, :3]
                sample_points.append(points)
                sample_labels.append(os.path.basename(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if sample_points:
            overview_path = os.path.join(output_dir, f"{subdir}_overview.png")
            visualizer.plot_comparison(
                sample_points, sample_labels,
                title=f"Dataset Overview - {subdir}",
                save_path=overview_path,
                ncols=3
            )
    
    print(f"Dataset overview saved to: {output_dir}")


def visualize_metrics(args, visualizer):
    """可视化评估指标"""
    print("Creating metrics visualizations...")
    
    # 查找指标文件
    metrics_files = glob.glob(os.path.join(args.input_dir, "*metrics*.json"))
    
    if not metrics_files:
        print("No metrics files found")
        return
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'metrics')
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {}
    
    # 加载所有指标文件
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            method_name = os.path.splitext(os.path.basename(metrics_file))[0]
            all_metrics[method_name] = metrics
            
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
            continue
    
    if all_metrics:
        # 创建指标对比图
        comparison_path = os.path.join(output_dir, 'metrics_comparison.png')
        visualizer.plot_metrics_comparison(
            all_metrics,
            title="Model Performance Comparison",
            save_path=comparison_path
        )
        
        print(f"Metrics comparison saved to: {comparison_path}")
    else:
        print("No valid metrics data found")


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"vis_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting visualization in {args.mode} mode")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # 设置颜色方案
    colors = {
        'default': {'sim': '#FF6B6B', 'real': '#4ECDC4', 'generated': '#45B7D1'},
        'pastel': {'sim': '#FFB3BA', 'real': '#BAFFC9', 'generated': '#BAE1FF'},
        'bright': {'sim': '#FF0000', 'real': '#00FF00', 'generated': '#0000FF'},
        'dark': {'sim': '#8B0000', 'real': '#006400', 'generated': '#000080'}
    }
    
    # 创建可视化器
    visualizer = PointCloudVisualizer(style=args.style, figsize=tuple(args.figure_size))
    if args.color_scheme in colors:
        visualizer.colors = colors[args.color_scheme]
    
    # 根据模式执行相应的可视化
    try:
        if args.mode == 'point_clouds':
            visualize_point_clouds(args, visualizer)
        
        elif args.mode == 'comparison':
            visualize_comparison(args, visualizer)
        
        elif args.mode == 'training_curves':
            visualize_training_curves(args, visualizer)
        
        elif args.mode == 'style_transfer':
            # 风格迁移特殊模式，结合对比和个体可视化
            visualize_comparison(args, visualizer)
            visualize_point_clouds(args, visualizer)
        
        elif args.mode == 'dataset_overview':
            visualize_dataset_overview(args, visualizer)
        
        elif args.mode == 'metrics':
            visualize_metrics(args, visualizer)
        
        print(f"\nVisualization completed! Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        return


if __name__ == "__main__":
    main()