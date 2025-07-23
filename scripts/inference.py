#!/usr/bin/env python3
"""
推理脚本
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

from config.config import Config
from models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
from models.pointnet2_encoder import ImprovedPointNet2Encoder
from models.chunk_fusion import ImprovedChunkFusion
from data.preprocessing import ImprovedPointCloudPreprocessor
from utils.visualization import PointCloudVisualizer
from utils.logger import Logger


class DiffusionInference:
    """Diffusion模型推理"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化日志
        self.logger = Logger(
            name='DiffusionInference',
            log_dir='logs/inference',
            file_output=True
        )
        
        # 加载配置和模型
        self.load_model(checkpoint_path)
        
        # 块融合模块
        self.chunk_fusion = ImprovedChunkFusion(overlap_ratio=self.config.overlap_ratio).to(self.device)
        
        # 预处理器
        self.preprocessor = ImprovedPointCloudPreprocessor(
            total_points=self.config.total_points,
            chunk_size=self.config.chunk_size,
            overlap_ratio=self.config.overlap_ratio
        )
        
        # 可视化器
        self.visualizer = PointCloudVisualizer()
        
        self.logger.info("Inference engine initialized successfully")
    
    def load_model(self, checkpoint_path: str):
        """加载模型"""
        self.logger.info(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # 初始化模型
        self.model = PointCloudDiffusionModel(
            input_dim=3,
            hidden_dims=[128, 256, 512, 1024],
            time_dim=self.config.time_embed_dim
        ).to(self.device)
        
        self.style_encoder = ImprovedPointNet2Encoder(
            input_channels=3,
            feature_dim=1024
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.style_encoder.load_state_dict(checkpoint['style_encoder_state_dict'])
        
        # 如果有EMA权重，使用EMA权重
        if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            self.logger.info("Using EMA weights")
            ema_params = checkpoint['ema_state_dict']['shadow_params']
            model_params = list(self.model.parameters())
            for param, ema_param in zip(model_params, ema_params):
                param.data.copy_(ema_param.data)
        
        self.model.eval()
        self.style_encoder.eval()
        
        # Diffusion过程
        self.diffusion_process = DiffusionProcess(
            num_timesteps=self.config.num_timesteps,
            beta_schedule=self.config.beta_schedule,
            device=self.device
        )
        
        self.logger.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    
    @torch.no_grad()
    def transfer_style(self, sim_points: np.ndarray, real_reference: np.ndarray,
                      use_ddim: bool = True, ddim_steps: int = 50) -> np.ndarray:
        """
        转换完整点云的风格
        Args:
            sim_points: 仿真点云 [120000, 3]
            real_reference: 真实参考点云 [N, 3]
            use_ddim: 是否使用DDIM加速采样
            ddim_steps: DDIM采样步数
        Returns:
            转换后的点云 [120000, 3]
        """
        start_time = time.time()
        
        # 1. 预处理和分块
        self.logger.info("Preprocessing and chunking point cloud...")
        sim_norm, sim_params = self.preprocessor.normalize_point_cloud(sim_points)
        sim_chunks = self.preprocessor.create_overlapping_chunks(sim_norm)
        
        self.logger.info(f"Created {len(sim_chunks)} chunks from {len(sim_points)} points")
        
        # 2. 提取风格特征
        self.logger.info("Extracting style features...")
        real_norm, _ = self.preprocessor.normalize_point_cloud(real_reference)
        
        # 如果参考点云太大，随机采样
        if len(real_norm) > self.config.chunk_size:
            indices = np.random.choice(len(real_norm), self.config.chunk_size, replace=False)
            real_sample = real_norm[indices]
        else:
            real_sample = real_norm
        
        real_tensor = torch.from_numpy(real_sample).float().unsqueeze(0).to(self.device)
        style_features = self.style_encoder(real_tensor).unsqueeze(1)  # [1, 1, feature_dim]
        
        # 3. 对每个块进行风格转换
        self.logger.info("Transferring style for each chunk...")
        transferred_chunks = []
        chunk_positions = []
        
        for i, (chunk_points, position) in enumerate(tqdm(sim_chunks, desc="Processing chunks")):
            chunk_tensor = torch.from_numpy(chunk_points).float().unsqueeze(0).to(self.device)
            
            # Diffusion采样
            shape = chunk_tensor.shape
            
            if use_ddim and ddim_steps < self.config.num_timesteps:
                # DDIM快速采样
                transferred_chunk = self.ddim_sample(
                    shape, style_features, ddim_steps
                )
            else:
                # 完整DDPM采样
                transferred_chunk = self.diffusion_process.sample(
                    self.model, shape, style_features
                )
            
            transferred_chunks.append(transferred_chunk.squeeze(0))
            chunk_positions.append(position)
            
            # 定期清理GPU内存
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 4. 融合所有块
        self.logger.info("Merging chunks into complete point cloud...")
        fused_points = self.chunk_fusion.merge_all_chunks(transferred_chunks, chunk_positions)
        
        # 5. 还原标准化
        final_points = self.preprocessor.denormalize_point_cloud(
            fused_points.cpu().numpy(),
            sim_params
        )
        
        # 确保输出点数正确
        if len(final_points) != len(sim_points):
            self.logger.warning(f"Point count mismatch: {len(final_points)} vs {len(sim_points)}")
            if len(final_points) > len(sim_points):
                final_points = final_points[:len(sim_points)]
            else:
                # 填充缺失的点
                repeat_indices = np.random.choice(len(final_points), len(sim_points) - len(final_points))
                final_points = np.vstack([final_points, final_points[repeat_indices]])
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Style transfer completed in {elapsed_time:.2f} seconds")
        
        return final_points
    
    def ddim_sample(self, shape, style_features, num_steps: int = 50) -> torch.Tensor:
        """DDIM快速采样"""
        device = self.device
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 选择时间步的子集
        timesteps = torch.linspace(
            self.config.num_timesteps - 1, 0, 
            num_steps, dtype=torch.long, device=device
        )
        
        for i in tqdm(range(len(timesteps) - 1), desc="DDIM sampling", leave=False):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            
            # 预测噪声
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.model(x, batch_t, style_features)
            
            # DDIM更新步骤
            alpha_t = self.diffusion_process.alphas_cumprod[t]
            alpha_t_prev = self.diffusion_process.alphas_cumprod[t_prev]
            
            # 预测x0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # 确定性更新（DDIM）
            x = torch.sqrt(alpha_t_prev) * x0_pred + \
                torch.sqrt(1 - alpha_t_prev) * predicted_noise
        
        return x
    
    def process_file(self, sim_path: str, real_reference_path: str, 
                    output_path: str, visualize: bool = False):
        """处理单个文件"""
        self.logger.info(f"Processing: {sim_path}")
        
        # 加载点云
        sim_points = np.load(sim_path).astype(np.float32)
        real_reference = np.load(real_reference_path).astype(np.float32)
        
        # 确保形状正确
        if sim_points.shape[1] != 3:
            sim_points = sim_points.T
        if real_reference.shape[1] != 3:
            real_reference = real_reference.T
        
        # 风格转换
        transferred = self.transfer_style(sim_points, real_reference)
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, transferred.astype(np.float32))
        self.logger.info(f"Saved result to: {output_path}")
        
        # 可视化
        if visualize:
            vis_path = output_path.replace('.npy', '_visualization.png')
            self.visualizer.plot_style_transfer_result(
                sim_points,
                transferred,
                real_reference,
                title='Point Cloud Style Transfer Result',
                save_path=vis_path,
                sample_size=5000
            )
            self.logger.info(f"Visualization saved to: {vis_path}")
        
        return transferred
    
    def process_folder(self, sim_folder: str, real_reference_path: str, 
                      output_folder: str, pattern: str = '*.npy',
                      visualize: bool = False):
        """批量处理文件夹"""
        os.makedirs(output_folder, exist_ok=True)
        
        # 加载真实参考
        real_reference = np.load(real_reference_path).astype(np.float32)
        if real_reference.shape[1] != 3:
            real_reference = real_reference.T
        
        # 查找所有匹配的文件
        sim_files = sorted(glob.glob(os.path.join(sim_folder, pattern)))
        
        if not sim_files:
            self.logger.warning(f"No files found matching pattern '{pattern}' in {sim_folder}")
            return
        
        self.logger.info(f"Found {len(sim_files)} files to process")
        
        # 处理每个文件
        for sim_file in tqdm(sim_files, desc="Processing files"):
            # 构建输出路径
            filename = os.path.basename(sim_file)
            output_path = os.path.join(output_folder, filename.replace('.npy', '_transferred.npy'))
            
            try:
                # 加载仿真点云
                sim_points = np.load(sim_file).astype(np.float32)
                if sim_points.shape[1] != 3:
                    sim_points = sim_points.T
                
                # 风格转换
                transferred = self.transfer_style(sim_points, real_reference)
                
                # 保存结果
                np.save(output_path, transferred.astype(np.float32))
                
                # 可视化第一个文件
                if visualize and sim_file == sim_files[0]:
                    vis_path = output_path.replace('.npy', '_visualization.png')
                    self.visualizer.plot_style_transfer_result(
                        sim_points,
                        transferred,
                        real_reference,
                        title=f'Style Transfer - {filename}',
                        save_path=vis_path,
                        sample_size=5000
                    )
                
            except Exception as e:
                self.logger.error(f"Failed to process {sim_file}: {e}")
                continue
        
        self.logger.info(f"Batch processing completed. Results saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Style Transfer Inference')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--sim_input', type=str, required=True, 
                       help='Simulation point cloud file or folder')
    parser.add_argument('--real_reference', type=str, required=True, 
                       help='Real reference point cloud')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output file or folder')
    
    # 可选参数
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--batch_process', action='store_true', 
                       help='Process entire folder')
    parser.add_argument('--pattern', type=str, default='*.npy', 
                       help='File pattern for batch processing')
    parser.add_argument('--use_ddim', action='store_true', 
                       help='Use DDIM for faster sampling')
    parser.add_argument('--ddim_steps', type=int, default=50, 
                       help='Number of DDIM steps')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = DiffusionInference(args.checkpoint, args.device)
    
    # 处理输入
    if args.batch_process or os.path.isdir(args.sim_input):
        # 批量处理
        inference.process_folder(
            args.sim_input, 
            args.real_reference, 
            args.output,
            args.pattern,
            args.visualize
        )
    else:
        # 单个文件
        inference.process_file(
            args.sim_input,
            args.real_reference,
            args.output,
            args.visualize
        )


if __name__ == "__main__":
    main()
