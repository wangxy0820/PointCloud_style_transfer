#!/usr/bin/env python3
"""
模型测试脚本
评估训练好的模型性能
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
from models.pointnet2_encoder import ImprovedPointNet2Encoder
from models.chunk_fusion import ImprovedChunkFusion
from data.dataset import PointCloudStyleTransferDataset
from evaluation.metrics import PointCloudMetrics
from evaluation.tester import DiffusionTester
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Test Point Cloud Style Transfer Model')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data directory')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Directory to save test results')
    parser.add_argument('--save_generated', action='store_true',
                       help='Save generated point clouds')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of samples to test (-1 for all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    # 评估选项
    parser.add_argument('--compute_all_metrics', action='store_true',
                       help='Compute all evaluation metrics')
    parser.add_argument('--metrics', nargs='+', 
                       default=['chamfer', 'emd', 'coverage'],
                       help='Metrics to compute')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存测试配置
    with open(os.path.join(output_dir, 'test_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading checkpoint...")
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']
    
    # 创建测试器
    tester = DiffusionTester(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=output_dir
    )
    
    # 创建测试数据集
    print("Loading test dataset...")
    test_dataset = PointCloudStyleTransferDataset(
        data_dir=args.test_data,
        split='test',
        chunk_size=config.chunk_size,
        augment=False
    )
    
    if args.num_samples > 0:
        # 限制测试样本数
        indices = list(range(min(args.num_samples, len(test_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Testing on {len(test_dataset)} samples...")
    
    # 运行测试
    results = tester.test(
        test_loader,
        compute_all_metrics=args.compute_all_metrics,
        save_generated=args.save_generated,
        save_visualizations=args.save_visualizations,
        metrics_to_compute=args.metrics
    )
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for metric_name, metric_value in results['average_metrics'].items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.6f}")
        else:
            print(f"{metric_name}: {metric_value}")
    
    print("="*60)
    
    # 保存详细结果
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
