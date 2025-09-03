# scripts/inference.py

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import sys
import time
from typing import Tuple
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
from data.preprocessing import PointCloudPreprocessor
from utils.logger import Logger

class PointCloudVisualizer:
    @staticmethod
    def visualize_comparison(original, reconstructed, reference, title='Comparison', save_path=None):
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(18, 6))
            
            def plot_ax(ax, points, title, cmap):
                sample_size = 8000
                if len(points) > sample_size:
                    indices = np.random.choice(len(points), sample_size, replace=False)
                    plot_points = points[indices]
                else:
                    plot_points = points
                ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], 
                           c=plot_points[:, 2], cmap=cmap, s=0.5)
                ax.set_title(title)
                ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
                ax.view_init(elev=20, azim=120)

            ax1 = fig.add_subplot(131, projection='3d')
            plot_ax(ax1, original, 'Original (Simulation)', 'viridis')
            
            ax2 = fig.add_subplot(132, projection='3d')
            plot_ax(ax2, reconstructed, 'Transferred', 'plasma')

            ax3 = fig.add_subplot(133, projection='3d')
            plot_ax(ax3, reference, 'Reference (Real)', 'coolwarm')
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except ImportError:
            print("Warning: matplotlib not available, skipping visualization")

class DiffusionInference:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        try:
            experiment_name = checkpoint_path.split(os.sep)[-2]
        except IndexError:
            experiment_name = "default_inference"

        self.logger = Logger(
            name='Inference',
            log_dir='logs/inference',
            experiment_name=experiment_name,
            file_output=True
        )
        
        self.config, self.model = self.load_model(checkpoint_path)
        
        self.diffusion_process = DiffusionProcess(self.config, device=str(self.device))
        self.preprocessor = PointCloudPreprocessor(
            total_points=self.config.total_points,
            global_points=self.config.global_points
        )
        self.logger.info("Hierarchical inference engine initialized.")

    def load_model(self, checkpoint_path: str) -> Tuple[Config, nn.Module]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        config = checkpoint['config']
        model = PointCloudDiffusionModel(config)
        
        if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
            self.logger.info("Loading EMA weights...")
            ema_shadow_params = checkpoint['ema_state_dict']['shadow_params']
            model_trainable_params = [p for p in model.parameters() if p.requires_grad]
            
            if len(ema_shadow_params) == len(model_trainable_params):
                
                with torch.no_grad():
                    for model_param, ema_param in zip(model_trainable_params, ema_shadow_params):
                        model_param.copy_(ema_param.to(self.device))
                
                self.logger.info("Successfully loaded EMA weights.")
            else:
                self.logger.error("EMA weights mismatch. Falling back to standard weights.")
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.logger.warning("EMA weights not found, loading standard model weights.")
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        self.logger.info("Model and config loaded successfully.")
        return config, model

    @torch.no_grad() # 为整个推理函数添加 no_grad 装饰器
    def transfer_style_hierarchical(self, source_points: np.ndarray, reference_points: np.ndarray, 
                                   num_steps: int, guidance_strength: float) -> np.ndarray:
        self.logger.info(f"Starting hierarchical style transfer for {source_points.shape[0]} points")
        start_time = time.time()
        
        source_norm, source_params = self.preprocessor.normalize_point_cloud(source_points)
        ref_norm, _ = self.preprocessor.normalize_point_cloud(reference_points)
        
        source_tensor = torch.from_numpy(source_norm).float().to(self.device).unsqueeze(0)
        ref_tensor = torch.from_numpy(ref_norm).float().to(self.device).unsqueeze(0)
        
        transferred = self.diffusion_process.guided_sample_loop(
            model=self.model,
            source_points=source_tensor,
            condition_points=ref_tensor,
            num_inference_steps=num_steps,
            guidance_strength=guidance_strength
        )
        
        transferred_norm = transferred.squeeze(0).cpu().numpy()
        transferred_points = self.preprocessor.denormalize_point_cloud(transferred_norm, source_params)
        
        self.logger.info(f"Hierarchical style transfer finished in {time.time() - start_time:.2f}s")
        return transferred_points

    def process_file(self, source_path: str, reference_path: str, output_path: str, 
                     visualize: bool, num_steps: int, guidance_strength: float):
        self.logger.info(f"Processing source: {source_path}")
        
        sim_points = np.loadtxt(source_path, delimiter=',') if source_path.endswith('.txt') else np.load(source_path)
        real_points = np.loadtxt(reference_path, delimiter=' ') if reference_path.endswith('.txt') else np.load(reference_path)
        
        self.logger.info(f"Using hierarchical processing for all inputs.")
        transferred_points = self.transfer_style_hierarchical(
            sim_points, real_points, num_steps, guidance_strength
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 保存为 .npy 格式以匹配输入
        np.save(output_path, transferred_points.astype(np.float32))
        self.logger.info(f"Transferred point cloud saved to: {output_path}")

        if visualize:
            vis_path = os.path.splitext(output_path)[0] + '.png'
            PointCloudVisualizer.visualize_comparison(
                original=sim_points,
                reconstructed=transferred_points,
                reference=real_points,
                title='Hierarchical Style Transfer Result',
                save_path=vis_path
            )
            self.logger.info(f"Visualization saved to: {vis_path}")

def main():
    parser = argparse.ArgumentParser(description='Hierarchical Point Cloud Style Transfer Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--source', type=str, required=True, help='Source (simulation) point cloud file')
    parser.add_argument('--reference', type=str, required=True, help='Reference (real) point cloud file for style')
    parser.add_argument('--output', type=str, required=True, help='Output file path for the transferred point cloud')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of DDIM inference steps')
    parser.add_argument('--guidance_strength', type=float, default=0.7, 
                        help='Guidance strength (0.0 to 1.0). Higher means more style transfer.')
    
    args = parser.parse_args()
    
    try:
        inference_engine = DiffusionInference(args.checkpoint, args.device)
        inference_engine.process_file(
            args.source, args.reference, args.output, 
            args.visualize, args.num_steps, args.guidance_strength
        )
        print("Inference completed successfully!")
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()