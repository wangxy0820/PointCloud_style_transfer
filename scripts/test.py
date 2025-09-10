#!/usr/bin/env python3

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
from data.dataset import HierarchicalPointCloudDataset
from evaluation.metrics import PointCloudMetrics
from utils.visualization import PointCloudVisualizer
from torch.utils.data import DataLoader


class Tester:
    """模型测试器"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', output_dir: str = 'test_results'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        self.load_model(checkpoint_path)
        
        # 评估器
        self.metrics = PointCloudMetrics(device=str(self.device))
        self.visualizer = PointCloudVisualizer()
        
    def load_model(self, checkpoint_path: str):
        """加载模型"""
        print(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # 初始化模型
        # MODIFIED: 现在直接从config初始化模型
        self.model = PointCloudDiffusionModel(self.config).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 如果有EMA权重，使用EMA权重
        if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            print("Using EMA weights")
            ema_params = checkpoint['ema_state_dict']['shadow_params']
            
            # 确保EMA参数加载到正确的设备
            ema_params = [p.to(self.device) for p in ema_params]
            
            model_params = list(self.model.parameters())
            # MODIFIED: 更安全地加载EMA权重
            num_to_load = min(len(model_params), len(ema_params))
            for i in range(num_to_load):
                model_params[i].data.copy_(ema_params[i].data)

        self.model.eval()
        
        # Diffusion过程
        # MODIFIED: 现在直接从config初始化
        self.diffusion_process = DiffusionProcess(
            config=self.config,
            device=str(self.device)
        )
        
        print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    
    @torch.no_grad()
    def test(self, test_loader, guidance_strength: float, num_inference_steps: int,
             compute_all_metrics: bool = True, save_generated: bool = False,
             save_visualizations: bool = False) -> dict:
        """
        测试模型
        """
        all_metrics = []
        
        # 创建保存目录
        if save_generated:
            gen_dir = os.path.join(self.output_dir, 'generated')
            os.makedirs(gen_dir, exist_ok=True)
        
        if save_visualizations:
            vis_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        # 测试循环
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            # MODIFIED: 从分层数据集中获取 'sim_full' 和 'real_full'
            sim_points = batch['sim_full'].to(self.device)
            real_points = batch['real_full'].to(self.device)
            
            batch_size = sim_points.shape[0]
            batch_metrics = {}
            
            # --- 1. Sim -> Real ---
            # MODIFIED: 直接调用 guided_sample_loop
            sim_to_real = self.diffusion_process.guided_sample_loop(
                model=self.model,
                source_points=sim_points,
                condition_points=real_points,
                num_inference_steps=num_inference_steps,
                guidance_strength=guidance_strength
            )
            
            # --- 2. Real -> Sim ---
            # MODIFIED: 直接调用 guided_sample_loop
            real_to_sim = self.diffusion_process.guided_sample_loop(
                model=self.model,
                source_points=real_points,
                condition_points=sim_points,
                num_inference_steps=num_inference_steps,
                guidance_strength=guidance_strength
            )
            
            # 计算指标
            if compute_all_metrics:
                cd_s2r = self.metrics.chamfer_distance(sim_to_real, real_points)
                batch_metrics['chamfer_sim_to_real'] = cd_s2r.mean().item()
                
                cd_r2s = self.metrics.chamfer_distance(real_to_sim, sim_points)
                batch_metrics['chamfer_real_to_sim'] = cd_r2s.mean().item()
                
                content_s2r = self.metrics.chamfer_distance(sim_to_real, sim_points)
                content_r2s = self.metrics.chamfer_distance(real_to_sim, real_points)
                batch_metrics['content_preservation'] = (content_s2r.mean().item() + content_r2s.mean().item()) / 2
            
            all_metrics.append(batch_metrics)
            
            # 保存生成的点云
            if save_generated:
                for i in range(batch_size):
                    idx = batch_idx * test_loader.batch_size + i
                    np.save(os.path.join(gen_dir, f'sim_to_real_{idx:04d}.npy'), sim_to_real[i].cpu().numpy())
                    np.save(os.path.join(gen_dir, f'real_to_sim_{idx:04d}.npy'), real_to_sim[i].cpu().numpy())
                    np.save(os.path.join(gen_dir, f'original_sim_{idx:04d}.npy'), sim_points[i].cpu().numpy())
                    np.save(os.path.join(gen_dir, f'original_real_{idx:04d}.npy'), real_points[i].cpu().numpy())
            
            # 保存可视化
            if save_visualizations and batch_idx < 5:
                for i in range(min(batch_size, 2)):
                    idx = batch_idx * test_loader.batch_size + i
                    self.visualizer.plot_style_transfer_result(
                        sim_points[i].cpu().numpy(), sim_to_real[i].cpu().numpy(), real_points[i].cpu().numpy(),
                        title=f'Test Sample {idx} - Sim to Real',
                        save_path=os.path.join(vis_dir, f'sample_{idx:04d}_s2r.png'))
                    self.visualizer.plot_style_transfer_result(
                        real_points[i].cpu().numpy(), real_to_sim[i].cpu().numpy(), sim_points[i].cpu().numpy(),
                        title=f'Test Sample {idx} - Real to Sim',
                        save_path=os.path.join(vis_dir, f'sample_{idx:04d}_r2s.png'))
        
        # 计算平均指标
        average_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    average_metrics[key] = np.mean(values)
        
        results = {'average_metrics': average_metrics}
        return results


def main():
    parser = argparse.ArgumentParser(description='Test Point Cloud Style Transfer Model')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save test results')
    parser.add_argument('--save_generated', action='store_true', help='Save generated point clouds')
    parser.add_argument('--save_visualizations', action='store_true', help='Save visualization images')
    
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to test (-1 for all)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # NEW: 添加推理参数控制
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of DDIM inference steps')
    parser.add_argument('--guidance_strength', type=float, default=0.7, help='Guidance strength (0.0 to 1.0)')

    parser.add_argument('--compute_all_metrics', action='store_true', help='Compute all evaluation metrics')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'test_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    tester = Tester(checkpoint_path=args.checkpoint, device=args.device, output_dir=output_dir)
    
    # MODIFIED: 数据集加载现在使用 `HierarchicalPointCloudDataset`
    print("Loading test dataset...")
    test_dataset = HierarchicalPointCloudDataset(
        processed_dir=args.test_data, # 直接指向包含 train/val/test 的根目录
        use_hierarchical=True # 确保加载完整数据
    )
    
    if args.num_samples > 0:
        indices = list(range(min(args.num_samples, len(test_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        # MODIFIED: 使用与训练时一致的collate_fn
        collate_fn=HierarchicalPointCloudDataset.hierarchical_collate_fn
    )
    
    print(f"Testing on {len(test_dataset)} samples...")
    results = tester.test(
        test_loader,
        guidance_strength=args.guidance_strength,
        num_inference_steps=args.num_inference_steps,
        compute_all_metrics=args.compute_all_metrics,
        save_generated=args.save_generated,
        save_visualizations=args.save_visualizations
    )
    
    print("\n" + "="*60 + "\nTEST RESULTS SUMMARY\n" + "="*60)
    for metric_name, metric_value in results['average_metrics'].items():
        print(f"{metric_name}: {metric_value:.6f}")
    print("="*60)
    
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: '<not serializable>')
    
    print(f"\nDetailed results saved to: {results_file}")

# NEW: 添加一个辅助collate_fn到Dataset类中以便调用
def hierarchical_collate_fn(batch):
    if not batch: return {}
    first_item = batch[0]
    result = {}
    tensor_keys = [k for k, v in first_item.items() if isinstance(v, torch.Tensor)]
    for key in tensor_keys:
        result[key] = torch.stack([item[key] for item in batch])
    
    other_keys = [k for k, v in first_item.items() if not isinstance(v, torch.Tensor)]
    for key in other_keys:
        result[key] = [item[key] for item in batch]
    return result

HierarchicalPointCloudDataset.hierarchical_collate_fn = staticmethod(hierarchical_collate_fn)


if __name__ == "__main__":
    main()