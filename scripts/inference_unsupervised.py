#!/usr/bin/env python3
"""
无监督模型推理脚本
"""

import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import glob
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_unsupervised import ConfigUnsupervised
from models.diffusion_model_unsupervised import UnsupervisedPointCloudDiffusionModel, UnsupervisedDiffusionProcess
from models.chunk_fusion import ImprovedChunkFusion
from data.preprocessing import PointCloudPreprocessor
from utils.visualization import PointCloudVisualizer
from utils.logger import Logger


class UnsupervisedDiffusionInference:
    """无监督Diffusion模型推理"""
    config = ConfigUnsupervised()
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化日志
        self.logger = Logger(
            name='UnsupervisedInference',
            log_dir='logs/inference',
            file_output=True
        )
        
        # 加载配置和模型
        self.load_model(checkpoint_path)
        
        # 块融合模块
        self.chunk_fusion = ImprovedChunkFusion(overlap_ratio=self.config.overlap_ratio).to(self.device)
        
        # 预处理器
        self.preprocessor = PointCloudPreprocessor(
            total_points=self.config.total_points,
            chunk_size=self.config.chunk_size,
            overlap_ratio=self.config.overlap_ratio
        )
        
        # 可视化器
        self.visualizer = PointCloudVisualizer()
        
        self.logger.info("Unsupervised inference engine initialized successfully")
    
    def load_model(self, checkpoint_path: str):
        """加载模型"""
        self.logger.info(f"Loading unsupervised model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # 初始化模型
        self.model = UnsupervisedPointCloudDiffusionModel(
            input_dim=3,
            hidden_dims=[128, 256, 512, 1024],
            time_dim=self.config.time_embed_dim,
            style_dim=256,
            content_dims=[64, 128, 256]
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 如果有EMA权重，使用EMA权重
        if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            self.logger.info("Using EMA weights")
            ema_params = checkpoint['ema_state_dict']['shadow_params']
            model_params = list(self.model.parameters())
            for param, ema_param in zip(model_params, ema_params):
                param.data.copy_(ema_param.data)
        
        self.model.eval()
        
        # Diffusion过程
        self.diffusion_process = UnsupervisedDiffusionProcess(
            num_timesteps=self.config.num_timesteps,
            beta_schedule=self.config.beta_schedule,
            device=self.device
        )
        
        self.logger.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    
    @torch.no_grad()
    def transfer_style(self, source_points: np.ndarray, reference_points: np.ndarray,
                      direction: str = 'auto', num_inference_steps: int = 50) -> np.ndarray:
        """
        转换点云风格（无监督版本）
        Args:
            source_points: 源点云 [N, 3]
            reference_points: 参考点云（用于提取目标风格） [M, 3]
            direction: 转换方向 ('sim_to_real', 'real_to_sim', 'auto')
            num_inference_steps: 推理步数
        Returns:
            转换后的点云 [N, 3]
        """
        start_time = time.time()
        
        # 1. 预处理源点云
        self.logger.info("Preprocessing source point cloud...")
        source_norm, source_params = self.preprocessor.normalize_point_cloud(source_points)
        source_chunks = self.preprocessor.create_overlapping_chunks(source_norm)
        
        self.logger.info(f"Created {len(source_chunks)} chunks from {len(source_points)} points")
        
        # 2. 提取参考风格
        self.logger.info("Extracting reference style...")
        ref_norm, _ = self.preprocessor.normalize_point_cloud(reference_points)
        
        # 如果参考点云太大，随机采样
        if len(ref_norm) > self.config.chunk_size:
            indices = np.random.choice(len(ref_norm), self.config.chunk_size, replace=False)
            ref_sample = ref_norm[indices]
        else:
            ref_sample = ref_norm
        
        ref_tensor = torch.from_numpy(ref_sample).float().unsqueeze(0).to(self.device)
        
        # 提取目标风格
        target_style = self.model.style_encoder(ref_tensor)
        
        # 3. 对每个块进行风格转换
        self.logger.info("Transferring style for each chunk...")
        transferred_chunks = []
        chunk_positions = []
        
        for i, (chunk_points, position) in enumerate(tqdm(source_chunks, desc="Processing chunks")):
            chunk_tensor = torch.from_numpy(chunk_points).float().unsqueeze(0).to(self.device)
            
            # 提取内容特征
            content_features = self.model.content_encoder(chunk_tensor)
            
            # 生成转换后的块
            shape = chunk_tensor.shape
            transferred_chunk = self.diffusion_process.sample(
                self.model,
                shape,
                style_condition=target_style,
                content_condition=content_features,
                num_inference_steps=num_inference_steps
            )
            
            transferred_chunks.append(transferred_chunk.squeeze(0))
            chunk_positions.append(position)
            
            # 定期清理GPU内存
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 4. 融合所有块
        self.logger.info("Merging chunks into complete point cloud...")
        fused_points = self.chunk_fusion.merge_all_chunks(transferred_chunks, chunk_positions)
        
        # 5. 逆归一化
        final_points = self.preprocessor.denormalize_point_cloud(
            fused_points.cpu().numpy(),
            source_params
        )
        
        # 确保输出点数正确
        if len(final_points) != len(source_points):
            self.logger.warning(f"Point count mismatch: {len(final_points)} vs {len(source_points)}")
            if len(final_points) > len(source_points):
                final_points = final_points[:len(source_points)]
            else:
                # 填充缺失的点
                repeat_indices = np.random.choice(len(final_points), len(source_points) - len(final_points))
                final_points = np.vstack([final_points, final_points[repeat_indices]])
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Style transfer completed in {elapsed_time:.2f} seconds")
        
        return final_points
    
    def process_file(self, source_path: str, reference_path: str, 
                    output_path: str, visualize: bool = False):
        """处理单个文件"""
        self.logger.info(f"Processing: {source_path}")
        self.logger.info(f"Reference: {reference_path}")
        
        # 加载点云
        source_points = np.load(source_path).astype(np.float32)
        reference_points = np.load(reference_path).astype(np.float32)
        
        # 确保形状正确
        if source_points.shape[1] != 3:
            source_points = source_points.T
        if reference_points.shape[1] != 3:
            reference_points = reference_points.T
        
        # 风格转换
        transferred = self.transfer_style(source_points, reference_points)
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, transferred.astype(np.float32))
        self.logger.info(f"Saved result to: {output_path}")
        
        # 可视化
        if visualize:
            vis_path = output_path.replace('.npy', '_visualization.png')
            
            # 为可视化采样点云
            sample_size = min(5000, len(source_points))
            indices = np.random.choice(len(source_points), sample_size, replace=False)
            
            self.visualizer.plot_style_transfer_result(
                source_points[indices],
                transferred[indices],
                reference_points[indices] if len(reference_points) >= sample_size else reference_points,
                title='Unsupervised Style Transfer Result',
                save_path=vis_path
            )
            self.logger.info(f"Visualization saved to: {vis_path}")
        
        return transferred


def main():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Style Transfer Inference')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--source', type=str, required=True, 
                       help='Source point cloud file')
    parser.add_argument('--reference', type=str, required=True, 
                       help='Reference point cloud for style')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output file path')
    
    # 可选参数
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--num_steps', type=int, default=50, 
                       help='Number of diffusion steps')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = UnsupervisedDiffusionInference(args.checkpoint, args.device)
    
    # 处理文件
    inference.process_file(
        args.source,
        args.reference,
        args.output,
        args.visualize
    )


if __name__ == "__main__":
    main()