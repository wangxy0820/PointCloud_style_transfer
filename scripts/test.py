#!/usr/bin/env python3
"""
点云风格迁移测试脚本
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.dataset import create_paired_data_loaders
from models.generator import CycleConsistentGenerator
from models.discriminator import HybridDiscriminator
from evaluation.metrics import PointCloudMetrics, ClassificationMetrics
from visualization.visualize import PointCloudVisualizer
import logging


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test Point Cloud Style Transfer Model')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test dataset directory')
    
    # 模型参数
    parser.add_argument('--chunk_size', type=int, default=8192,
                       help='Point cloud chunk size')
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Latent dimension size')
    parser.add_argument('--generator_dim', type=int, default=256,
                       help='Generator style dimension')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for testing')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Directory to save test results')
    parser.add_argument('--save_generated', action='store_true',
                       help='Save generated point clouds')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                       help='Number of samples to visualize')
    
    # 评估选项
    parser.add_argument('--compute_all_metrics', action='store_true',
                       help='Compute all available metrics')
    parser.add_argument('--save_metrics_csv', action='store_true',
                       help='Save metrics to CSV file')
    
    # 测试模式
    parser.add_argument('--test_split', type=str, default='test',
                       choices=['test', 'val', 'train'],
                       help='Which data split to test on')
    parser.add_argument('--direction', type=str, default='sim2real',
                       choices=['sim2real', 'real2sim', 'both'],
                       help='Transfer direction to test')
    
    return parser.parse_args()


def load_model(model_path: str, config: Config, device: torch.device):
    """
    加载训练好的模型
    Args:
        model_path: 模型检查点路径
        config: 配置对象
        device: 设备
    Returns:
        加载的生成器模型
    """
    # 创建模型
    generator = CycleConsistentGenerator(
        input_channels=config.input_dim,
        feature_channels=config.pointnet_channels,
        style_dim=config.generator_dim,
        latent_dim=config.latent_dim,
        num_points=config.chunk_size
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    return generator


def run_inference(generator: torch.nn.Module, data_loader, device: torch.device, 
                 direction: str = 'sim2real') -> dict:
    """
    运行推理并收集结果
    Args:
        generator: 生成器模型
        data_loader: 数据加载器
        device: 设备
        direction: 转换方向
    Returns:
        推理结果字典
    """
    results = {
        'original': [],
        'generated': [],
        'reference': [],
        'file_paths': []
    }
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Running {direction} inference'):
            sim_points = batch['sim_points'].to(device)
            real_points = batch['real_points'].to(device)
            
            if direction == 'sim2real':
                generated = generator.sim2real(sim_points, real_points)
                results['original'].append(sim_points.cpu())
                results['generated'].append(generated.cpu())
                results['reference'].append(real_points.cpu())
                
            elif direction == 'real2sim':
                generated = generator.real2sim(real_points, sim_points)
                results['original'].append(real_points.cpu())
                results['generated'].append(generated.cpu())
                results['reference'].append(sim_points.cpu())
                
            elif direction == 'both':
                fake_real, fake_sim = generator(sim_points, real_points)
                results['original'].extend([sim_points.cpu(), real_points.cpu()])
                results['generated'].extend([fake_real.cpu(), fake_sim.cpu()])
                results['reference'].extend([real_points.cpu(), sim_points.cpu()])
            
            # 保存文件路径信息
            if 'sim_file' in batch:
                results['file_paths'].extend(batch['sim_file'])
            if 'real_file' in batch:
                results['file_paths'].extend(batch['real_file'])
    
    # 合并所有批次
    results['original'] = torch.cat(results['original'], dim=0)
    results['generated'] = torch.cat(results['generated'], dim=0)
    results['reference'] = torch.cat(results['reference'], dim=0)
    
    return results


def compute_metrics(results: dict, metrics_calculator: PointCloudMetrics) -> dict:
    """
    计算评估指标
    Args:
        results: 推理结果
        metrics_calculator: 指标计算器
    Returns:
        计算的指标字典
    """
    original = results['original']
    generated = results['generated']
    reference = results['reference']
    
    print("Computing metrics...")
    
    # 计算生成质量指标
    generation_metrics = metrics_calculator.compute_all_metrics(generated, reference)
    
    # 计算保持内容一致性指标
    content_metrics = metrics_calculator.compute_all_metrics(generated, original)
    
    # 重命名指标以区分
    final_metrics = {}
    for k, v in generation_metrics.items():
        final_metrics[f'generation_{k}'] = v
    
    for k, v in content_metrics.items():
        final_metrics[f'content_preservation_{k}'] = v
    
    # 计算额外的风格迁移特定指标
    final_metrics['style_transfer_ratio'] = compute_style_transfer_ratio(
        original, generated, reference
    )
    
    return final_metrics


def compute_style_transfer_ratio(original: torch.Tensor, generated: torch.Tensor, 
                               reference: torch.Tensor) -> float:
    """
    计算风格迁移比率
    Args:
        original: 原始点云
        generated: 生成点云
        reference: 参考点云
    Returns:
        风格迁移比率
    """
    # 计算生成点云到参考点云的距离
    gen_to_ref_dist = torch.cdist(generated, reference).min(dim=2)[0].mean()
    
    # 计算原始点云到参考点云的距离
    orig_to_ref_dist = torch.cdist(original, reference).min(dim=2)[0].mean()
    
    # 风格迁移比率 = 距离改善程度
    if orig_to_ref_dist > 0:
        ratio = 1.0 - (gen_to_ref_dist / orig_to_ref_dist)
        return max(0.0, ratio.item())
    else:
        return 0.0


def save_results(results: dict, metrics: dict, output_dir: str, args):
    """
    保存测试结果
    Args:
        results: 推理结果
        metrics: 评估指标
        output_dir: 输出目录
        args: 命令行参数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标到JSON文件
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to: {metrics_file}")
    
    # 保存指标到CSV文件（如果需要）
    if args.save_metrics_csv:
        import pandas as pd
        df = pd.DataFrame([metrics])
        csv_file = os.path.join(output_dir, 'metrics.csv')
        df.to_csv(csv_file, index=False)
        print(f"Metrics CSV saved to: {csv_file}")
    
    # 保存生成的点云（如果需要）
    if args.save_generated:
        generated_dir = os.path.join(output_dir, 'generated_point_clouds')
        os.makedirs(generated_dir, exist_ok=True)
        
        print("Saving generated point clouds...")
        for i in tqdm(range(len(results['generated']))):
            # 保存为.npy文件
            np.save(
                os.path.join(generated_dir, f'generated_{i:04d}.npy'),
                results['generated'][i].numpy()
            )
            np.save(
                os.path.join(generated_dir, f'original_{i:04d}.npy'),
                results['original'][i].numpy()
            )
            np.save(
                os.path.join(generated_dir, f'reference_{i:04d}.npy'),
                results['reference'][i].numpy()
            )
    
    # 生成可视化（如果需要）
    if args.save_visualizations:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        visualizer = PointCloudVisualizer()
        
        print("Generating visualizations...")
        num_samples = min(args.num_vis_samples, len(results['generated']))
        
        for i in tqdm(range(num_samples), desc="Creating visualizations"):
            # 单独可视化
            visualizer.save_point_cloud(
                results['original'][i].numpy(),
                os.path.join(vis_dir, f'sample_{i:03d}_original.png'),
                title=f'Original - Sample {i+1}',
                color='sim'
            )
            
            visualizer.save_point_cloud(
                results['generated'][i].numpy(),
                os.path.join(vis_dir, f'sample_{i:03d}_generated.png'),
                title=f'Generated - Sample {i+1}',
                color='generated'
            )
            
            visualizer.save_point_cloud(
                results['reference'][i].numpy(),
                os.path.join(vis_dir, f'sample_{i:03d}_reference.png'),
                title=f'Reference - Sample {i+1}',
                color='real'
            )
            
            # 对比可视化
            visualizer.plot_style_transfer_result(
                results['original'][i].numpy(),
                results['generated'][i].numpy(),
                results['reference'][i].numpy(),
                title=f'Style Transfer Result - Sample {i+1}',
                save_path=os.path.join(vis_dir, f'sample_{i:03d}_comparison.png')
            )


def print_metrics_summary(metrics: dict):
    """打印指标摘要"""
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    
    # 主要指标
    main_metrics = {
        'Generation Quality': {
            'Chamfer Distance': metrics.get('generation_chamfer_distance', 'N/A'),
            'EMD': metrics.get('generation_earth_mover_distance', 'N/A'),
            'Hausdorff Distance': metrics.get('generation_hausdorff_distance', 'N/A'),
        },
        'Content Preservation': {
            'Chamfer Distance': metrics.get('content_preservation_chamfer_distance', 'N/A'),
            'Coverage Score': metrics.get('generation_coverage_score', 'N/A'),
        },
        'Style Transfer': {
            'Transfer Ratio': metrics.get('style_transfer_ratio', 'N/A'),
            'Uniformity Score': metrics.get('generation_uniformity_score', 'N/A'),
        }
    }
    
    for category, category_metrics in main_metrics.items():
        print(f"\n{category}:")
        print("-" * 30)
        for metric_name, value in category_metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.6f}")
            else:
                print(f"  {metric_name}: {value}")
    
    print("="*60)


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'test.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting test with model: {args.model_path}")
    logger.info(f"Test data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # 创建配置
    config = Config()
    config.chunk_size = args.chunk_size
    config.latent_dim = args.latent_dim
    config.generator_dim = args.generator_dim
    config.device = str(device)
    
    # 加载模型
    print("Loading model...")
    try:
        generator = load_model(args.model_path, config, device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # 创建数据加载器
    print("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_paired_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment_train=False  # 测试时不使用数据增强
        )
        
        # 选择测试数据
        if args.test_split == 'train':
            data_loader = train_loader
        elif args.test_split == 'val':
            data_loader = val_loader
        else:
            data_loader = test_loader
        
        logger.info(f"Testing on {args.test_split} split with {len(data_loader)} batches")
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # 运行推理
    print(f"Running inference in {args.direction} direction...")
    try:
        results = run_inference(generator, data_loader, device, args.direction)
        logger.info(f"Inference completed on {len(results['generated'])} samples")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return
    
    # 计算指标
    if args.compute_all_metrics:
        print("Computing evaluation metrics...")
        try:
            metrics_calculator = PointCloudMetrics(device=str(device))
            metrics = compute_metrics(results, metrics_calculator)
            logger.info("Metrics computation completed")
        except Exception as e:
            logger.error(f"Metrics computation failed: {e}")
            metrics = {}
    else:
        metrics = {}
    
    # 保存结果
    print("Saving results...")
    try:
        save_results(results, metrics, output_dir, args)
        logger.info(f"Results saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    # 打印结果摘要
    if metrics:
        print_metrics_summary(metrics)
    
    print(f"\nTest completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()