#!/usr/bin/env python3
"""
最终测试脚本 - 确保一切正常工作
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_everything():
    """测试所有组件"""
    print("="*60)
    print("FINAL COMPREHENSIVE TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 测试模型创建
    print("\n1. Testing model creation...")
    try:
        from models.diffusion_model import PointCloudDiffusionModel, DiffusionProcess
        from models.pointnet2_encoder import ImprovedPointNet2Encoder
        
        # 创建组件
        style_encoder = ImprovedPointNet2Encoder(
            input_channels=3,
            feature_dim=1024
        ).to(device)
        
        diffusion_model = PointCloudDiffusionModel(
            input_dim=3,
            hidden_dims=[128, 256, 512, 1024],
            time_dim=256,
            context_dim=1024,
            num_heads=8
        ).to(device)
        
        diffusion_process = DiffusionProcess(
            num_timesteps=1000,
            beta_schedule='cosine',
            device=str(device)
        )
        
        print("✓ All models created successfully!")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # 2. 测试前向传播
    print("\n2. Testing forward pass...")
    try:
        # 测试数据
        batch_size = 2
        num_points = 2048
        sim_points = torch.randn(batch_size, num_points, 3).to(device)
        real_points = torch.randn(batch_size, num_points, 3).to(device)
        
        # 提取风格特征
        style_encoder.eval()
        with torch.no_grad():
            style_features = style_encoder(real_points)
            print(f"  Style features shape: {style_features.shape}")
            style_features = style_features.unsqueeze(1)
            print(f"  Style features (with seq dim): {style_features.shape}")
        
        # 添加噪声
        t = torch.randint(0, 1000, (batch_size,), device=device)
        noise = torch.randn_like(sim_points)
        noisy_points = diffusion_process.q_sample(sim_points, t, noise)
        print(f"  Noisy points shape: {noisy_points.shape}")
        
        # 预测噪声
        diffusion_model.eval()
        with torch.no_grad():
            predicted_noise = diffusion_model(noisy_points, t, style_features)
            print(f"  Predicted noise shape: {predicted_noise.shape}")
        
        print("✓ Forward pass successful!")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试训练步骤
    print("\n3. Testing training step...")
    try:
        from config.config import Config
        from training.trainer import DiffusionTrainer
        
        config = Config()
        config.batch_size = 2
        config.num_workers = 0
        
        # 创建训练器
        trainer = DiffusionTrainer(config)
        
        # 模拟批次数据
        batch = {
            'sim_points': torch.randn(2, 2048, 3).to(device),
            'real_points': torch.randn(2, 2048, 3).to(device),
            'sim_position': (0, 2048),
            'real_position': (0, 2048),
            'chunk_idx': 0,
            'num_chunks': 10,
            'file_idx': 0,
            'norm_params': {'sim': {}, 'real': {}}
        }
        
        # 训练步骤
        losses = trainer.train_step(batch)
        
        print("✓ Training step successful!")
        print("  Losses:")
        for k, v in losses.items():
            print(f"    {k}: {v:.6f}")
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试真实数据
    print("\n4. Testing with real data...")
    try:
        from data.dataset import create_dataloaders
        
        train_loader, _, _ = create_dataloaders(
            data_dir='datasets/processed',
            batch_size=2,
            num_workers=0,
            chunk_size=2048
        )
        
        # 获取一个真实批次
        real_batch = next(iter(train_loader))
        real_batch['sim_points'] = real_batch['sim_points'].to(device)
        real_batch['real_points'] = real_batch['real_points'].to(device)
        
        # 测试
        losses = trainer.train_step(real_batch)
        
        print("✓ Real data test successful!")
        print("  Losses:")
        for k, v in losses.items():
            print(f"    {k}: {v:.6f}")
        
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("Running final comprehensive test...\n")
    
    success = test_everything()
    
    print("\n" + "="*60)
    if success:
        print("✓ ALL TESTS PASSED! The model is ready for training.")
        print("\nYou can now run:")
        print("  python scripts/train.py --data_dir datasets/processed --experiment_name my_experiment --batch_size 8 --num_epochs 40")
    else:
        print("✗ Tests failed. Please check the errors above.")
    print("="*60)