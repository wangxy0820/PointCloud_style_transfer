#!/usr/bin/env python3
"""
点云风格迁移推理脚本
用于对新的点云数据进行风格迁移
"""

import argparse
import os
import sys
import torch
import numpy as np
import glob
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models.generator import CycleConsistentGenerator
from data.preprocess import PointCloudPreprocessor
from visualization.visualize import PointCloudVisualizer
import logging


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Point Cloud Style Transfer Inference')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input point clouds (.npy files)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save generated point clouds')
    
    # 风格参考
    parser.add_argument('--style_reference', type=str, default='',
                       help='Path to style reference point cloud (.npy file)')
    parser.add_argument('--style_dir', type=str, default='',
                       help='Directory containing style reference point clouds')
    
    # 模型参数
    parser.add_argument('--chunk_size', type=int, default=8192,
                       help='Point cloud chunk size')
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Latent dimension size')
    parser.add_argument('--generator_dim', type=int, default=256,
                       help='Generator style dimension')
    
    # 推理参数
    parser.add_argument('--direction', type=str, default='sim2real',
                       choices=['sim2real', 'real2sim'],
                       help='Transfer direction')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    # 预处理参数
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess input point clouds (chunking)')
    parser.add_argument('--chunk_method', type=str, default='spatial',
                       choices=['spatial', 'random', 'sliding'],
                       help='Chunking method for preprocessing')
    parser.add_argument('--merge_chunks', action='store_true',
                       help='Merge chunks back to full point cloud')
    
    # 输出选项
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Save intermediate chunk results')
    parser.add_argument('--create_visualization', action='store_true',
                       help='Create visualization images')
    parser.add_argument('--save_comparison', action='store_true',
                       help='Save before/after comparison images')
    
    # 其他选项
    parser.add_argument('--random_style', action='store_true',
                       help='Use random style for each input')
    parser.add_argument('--output_format', type=str, default='npy',
                       choices=['npy', 'ply', 'both'],
                       help='Output file format')
    
    return parser.parse_args()


def load_model(model_path: str, config: Config, device: torch.device):
    """加载训练好的模型"""
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


def load_point_cloud(file_path: str) -> np.ndarray:
    """加载点云文件"""
    if file_path.endswith('.npy'):
        points = np.load(file_path)
    elif file_path.endswith('.ply'):
        # 这里可以添加PLY文件加载代码
        raise NotImplementedError("PLY format not implemented yet")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # 确保是3D点云
    if points.shape[1] > 3:
        points = points[:, :3]
    
    return points.astype(np.float32)


def save_point_cloud(points: np.ndarray, file_path: str, format: str = 'npy'):
    """保存点云文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if format == 'npy':
        np.save(file_path, points)
    elif format == 'ply':
        # 这里可以添加PLY文件保存代码
        raise NotImplementedError("PLY format not implemented yet")
    elif format == 'both':
        # 保存两种格式
        base_path = os.path.splitext(file_path)[0]
        np.save(base_path + '.npy', points)
        # save_ply(points, base_path + '.ply')


def preprocess_point_cloud(points: np.ndarray, chunk_size: int, 
                         chunk_method: str = 'spatial') -> list:
    """预处理点云（分块）"""
    preprocessor = PointCloudPreprocessor(chunk_size=chunk_size)
    
    # 标准化
    points = preprocessor.normalize_point_cloud(points)
    
    # 分块
    if chunk_method == 'spatial':
        num_chunks = max(1, len(points) // chunk_size)
        chunks = preprocessor.spatial_chunk(points, num_chunks)
    elif chunk_method == 'sliding':
        chunks = preprocessor.sliding_window_chunk(points)
    else:  # random
        num_chunks = max(1, len(points) // chunk_size)
        chunks = []
        for i in range(num_chunks):
            chunk = preprocessor.random_chunk(points, chunk_size)
            chunks.append(preprocessor.normalize_point_cloud(chunk))
    
    return chunks


def merge_chunks(chunks: list, original_size: int) -> np.ndarray:
    """合并分块回完整点云"""
    # 简单的合并策略：连接所有块并随机采样到原始大小
    all_points = np.concatenate(chunks, axis=0)
    
    if len(all_points) > original_size:
        # 随机采样
        indices = np.random.choice(len(all_points), original_size, replace=False)
        return all_points[indices]
    else:
        # 如果点数不足，进行上采样
        indices = np.random.choice(len(all_points), original_size, replace=True)
        return all_points[indices]


def inference_single_file(generator: torch.nn.Module, 
                         input_path: str,
                         style_reference: torch.Tensor,
                         device: torch.device,
                         args) -> np.ndarray:
    """对单个文件进行推理"""
    # 加载输入点云
    input_points = load_point_cloud(input_path)
    original_size = len(input_points)
    
    # 预处理（如果需要）
    if args.preprocess:
        chunks = preprocess_point_cloud(input_points, args.chunk_size, args.chunk_method)
        generated_chunks = []
        
        with torch.no_grad():
            for chunk in chunks:
                # 转换为张量
                chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
                style_tensor = style_reference.unsqueeze(0)
                
                # 风格迁移
                if args.direction == 'sim2real':
                    generated_chunk = generator.sim2real(chunk_tensor, style_tensor)
                else:
                    generated_chunk = generator.real2sim(chunk_tensor, style_tensor)
                
                generated_chunks.append(generated_chunk.squeeze(0).cpu().numpy())
        
        # 合并分块
        if args.merge_chunks:
            generated_points = merge_chunks(generated_chunks, original_size)
        else:
            generated_points = np.concatenate(generated_chunks, axis=0)
    
    else:
        # 直接处理整个点云（需要确保大小合适）
        if len(input_points) != args.chunk_size:
            preprocessor = PointCloudPreprocessor(chunk_size=args.chunk_size)
            if len(input_points) > args.chunk_size:
                input_points = preprocessor.random_chunk(input_points, args.chunk_size)
            else:
                # 上采样
                indices = np.random.choice(len(input_points), args.chunk_size, replace=True)
                input_points = input_points[indices]
        
        # 标准化
        preprocessor = PointCloudPreprocessor()
        input_points = preprocessor.normalize_point_cloud(input_points)
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_points).unsqueeze(0).to(device)
            style_tensor = style_reference.unsqueeze(0)
            
            if args.direction == 'sim2real':
                generated = generator.sim2real(input_tensor, style_tensor)
            else:
                generated = generator.real2sim(input_tensor, style_tensor)
            
            generated_points = generated.squeeze(0).cpu().numpy()
    
    return generated_points


def load_style_references(args, device: torch.device) -> list:
    """加载风格参考点云"""
    style_references = []
    
    if args.style_reference:
        # 单个风格参考文件
        style_points = load_point_cloud(args.style_reference)
        preprocessor = PointCloudPreprocessor()
        style_points = preprocessor.normalize_point_cloud(style_points)
        
        # 调整大小
        if len(style_points) != args.chunk_size:
            if len(style_points) > args.chunk_size:
                style_points = preprocessor.random_chunk(style_points, args.chunk_size)
            else:
                indices = np.random.choice(len(style_points), args.chunk_size, replace=True)
                style_points = style_points[indices]
        
        style_tensor = torch.from_numpy(style_points).to(device)
        style_references.append(style_tensor)
    
    elif args.style_dir:
        # 多个风格参考文件
        style_files = glob.glob(os.path.join(args.style_dir, "*.npy"))
        
        for style_file in style_files[:10]:  # 最多加载10个风格参考
            style_points = load_point_cloud(style_file)
            preprocessor = PointCloudPreprocessor()
            style_points = preprocessor.normalize_point_cloud(style_points)
            
            # 调整大小
            if len(style_points) != args.chunk_size:
                if len(style_points) > args.chunk_size:
                    style_points = preprocessor.random_chunk(style_points, args.chunk_size)
                else:
                    indices = np.random.choice(len(style_points), args.chunk_size, replace=True)
                    style_points = style_points[indices]
            
            style_tensor = torch.from_numpy(style_points).to(device)
            style_references.append(style_tensor)
    
    else:
        # 使用随机风格（创建一个随机点云作为风格参考）
        random_points = np.random.randn(args.chunk_size, 3).astype(np.float32)
        style_tensor = torch.from_numpy(random_points).to(device)
        style_references.append(style_tensor)
    
    return style_references


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
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, 'inference.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting inference with model: {args.model_path}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Transfer direction: {args.direction}")
    
    # 创建配置
    config = Config()
    config.chunk_size = args.chunk_size
    config.latent_dim = args.latent_dim
    config.generator_dim = args.generator_dim
    
    # 加载模型
    print("Loading model...")
    try:
        generator = load_model(args.model_path, config, device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # 加载风格参考
    print("Loading style references...")
    try:
        style_references = load_style_references(args, device)
        logger.info(f"Loaded {len(style_references)} style references")
    except Exception as e:
        logger.error(f"Failed to load style references: {e}")
        return
    
    # 获取输入文件列表
    input_files = glob.glob(os.path.join(args.input_dir, "*.npy"))
    if not input_files:
        logger.error(f"No .npy files found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(input_files)} input files")
    
    # 创建可视化器（如果需要）
    if args.create_visualization:
        visualizer = PointCloudVisualizer()
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # 处理每个输入文件
    print("Processing input files...")
    for i, input_file in enumerate(tqdm(input_files, desc="Processing files")):
        try:
            # 选择风格参考
            if args.random_style:
                style_ref = style_references[np.random.randint(len(style_references))]
            else:
                style_ref = style_references[i % len(style_references)]
            
            # 进行推理
            generated_points = inference_single_file(
                generator, input_file, style_ref, device, args
            )
            
            # 保存结果
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(args.output_dir, f"{base_name}_generated.npy")
            
            save_point_cloud(generated_points, output_file, args.output_format)
            
            # 创建可视化（如果需要）
            if args.create_visualization:
                # 原始点云可视化
                original_points = load_point_cloud(input_file)
                
                vis_file = os.path.join(vis_dir, f"{base_name}_generated.png")
                visualizer.save_point_cloud(
                    generated_points, vis_file,
                    title=f"Generated - {base_name}",
                    color='generated'
                )
                
                # 对比可视化（如果需要）
                if args.save_comparison:
                    comparison_file = os.path.join(vis_dir, f"{base_name}_comparison.png")
                    style_points = style_ref.cpu().numpy()
                    
                    visualizer.plot_style_transfer_result(
                        original_points[:args.chunk_size] if len(original_points) > args.chunk_size else original_points,
                        generated_points,
                        style_points,
                        title=f"Style Transfer - {base_name}",
                        save_path=comparison_file
                    )
            
            logger.info(f"Processed: {base_name}")
            
        except Exception as e:
            logger.error(f"Failed to process {input_file}: {e}")
            continue
    
    logger.info(f"Inference completed. Results saved to: {args.output_dir}")
    print(f"\nInference completed! Generated files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()