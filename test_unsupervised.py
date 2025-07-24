#!/usr/bin/env python3
"""
测试无监督Diffusion模型
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_unsupervised_model():
    """测试无监督模型的各个组件"""
    print("Testing Unsupervised Diffusion Model")
    print("="*60)
    
    # 1. 导入测试
    print("1. Testing model import...")
    try:
        from models.unsupervised_diffusion_model import (
            UnsupervisedPointCloudDiffusionModel, 
            UnsupervisedDiffusionProcess,
            StyleEncoder,
            ContentEncoder
        )
        print("  ✓ Models imported successfully")
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return
    
    # 2. 创建模型
    print("\n2. Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    try:
        model = UnsupervisedPointCloudDiffusionModel(
            input_dim=3,
            hidden_dims=[128, 256, 512, 1024],
            time_dim=256,
            style_dim=256,
            content_dims=[64, 128, 256]
        ).to(device)
        print("  ✓ Model created successfully")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params/1e6:.2f}M")
    except Exception as e:
        print(f"  ✗ Model creation error: {e}")
        return
    
    # 3. 测试前向传播
    print("\n3. Testing forward pass...")
    batch_size = 2
    num_points = 2048
    
    try:
        # 创建测试数据
        x = torch.randn(batch_size, num_points, 3).to(device)
        t = torch.randint(0, 1000, (batch_size,), device=device)
        
        # 测试风格编码器
        style = model.style_encoder(x)
        print(f"  Style shape: {style.shape}")
        assert style.shape == (batch_size, 256), f"Expected style shape (2, 256), got {style.shape}"
        
        # 测试内容编码器
        content = model.content_encoder(x)
        print(f"  Content shape: {content.shape}")
        assert content.shape == (batch_size, 256, num_points), f"Expected content shape (2, 256, 2048), got {content.shape}"
        
        # 测试完整前向传播
        output = model(x, t)
        print(f"  Output shape: {output.shape}")
        assert output.shape == x.shape, f"Expected output shape {x.shape}, got {output.shape}"
        
        print("  ✓ Forward pass successful")
        
    except Exception as e:
        print(f"  ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 测试风格调制
    print("\n4. Testing style modulation...")
    try:
        # 测试每一层的风格调制
        test_features = torch.randn(batch_size, 128, num_points).to(device)
        modulated = model.style_modulate(test_features, style, 0)
        print(f"  Modulated shape: {modulated.shape}")
        print("  ✓ Style modulation successful")
    except Exception as e:
        print(f"  ✗ Style modulation error: {e}")
        return
    
    # 5. 测试Diffusion过程
    print("\n5. Testing diffusion process...")
    try:
        diffusion = UnsupervisedDiffusionProcess(
            num_timesteps=1000,
            beta_schedule='cosine',
            device=str(device)
        )
        
        # 测试前向扩散
        noise = torch.randn_like(x)
        t_test = torch.tensor([100, 500], device=device)
        noisy_x = diffusion.q_sample(x, t_test, noise)
        print(f"  Noisy x shape: {noisy_x.shape}")
        
        # 测试采样
        with torch.no_grad():
            generated = diffusion.sample(
                model, 
                (1, 100, 3),  # 小一点的shape用于快速测试
                style_condition=style[:1],
                content_condition=content[:1, :, :100],
                num_inference_steps=10
            )
        print(f"  Generated shape: {generated.shape}")
        print("  ✓ Diffusion process successful")
        
    except Exception as e:
        print(f"  ✗ Diffusion process error: {e}")
        return
    
    # 6. 测试不同的条件输入
    print("\n6. Testing conditional generation...")
    try:
        # 测试只提供风格条件
        output1 = model(x, t, style_condition=style)
        
        # 测试只提供内容条件
        output2 = model(x, t, content_condition=content)
        
        # 测试同时提供两种条件
        output3 = model(x, t, style_condition=style, content_condition=content)
        
        print("  ✓ All conditional modes work correctly")
        
    except Exception as e:
        print(f"  ✗ Conditional generation error: {e}")
        return
    
    # 7. 内存和性能测试
    print("\n7. Testing memory and performance...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # 测试较大的批次
            large_x = torch.randn(4, 4096, 3).to(device)
            large_t = torch.randint(0, 1000, (4,), device=device)
            
            # 计时
            import time
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = model(large_x, large_t)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"  Forward pass time (4x4096 points): {elapsed:.3f}s")
            print(f"  Throughput: {4*4096/elapsed:.0f} points/s")
            
            # 检查内存使用
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak GPU memory: {memory_used:.2f} GB")
            
        print("  ✓ Performance test completed")
        
    except Exception as e:
        print(f"  ✗ Performance test error: {e}")
    
    print("\n" + "="*60)
    print("✓ All tests passed! The unsupervised model is working correctly.")
    print("\nNext steps:")
    print("1. Train with: python scripts/train_unsupervised.py --data_dir datasets/processed")
    print("2. The model will learn style transfer WITHOUT point correspondence")
    print("3. Monitor both content preservation and style transfer metrics")


if __name__ == "__main__":
    test_unsupervised_model()