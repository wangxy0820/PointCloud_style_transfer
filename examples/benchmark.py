"""
性能基准测试脚本
"""

import torch
import time
import numpy as np
from memory_profiler import profile

from models.diffusion_model import PointCloudDiffusionModel
from models.pointnet2_encoder import ImprovedPointNet2Encoder


@profile
def benchmark_model_memory():
    """测试模型内存使用"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = PointCloudDiffusionModel().to(device)
    encoder = ImprovedPointNet2Encoder().to(device)
    
    # 不同批大小的测试
    batch_sizes = [1, 2, 4, 8, 16]
    chunk_size = 2048
    
    for bs in batch_sizes:
        try:
            # 创建输入
            points = torch.randn(bs, chunk_size, 3).to(device)
            t = torch.randint(0, 1000, (bs,)).to(device)
            style_features = torch.randn(bs, 1, 1024).to(device)
            
            # 前向传播
            with torch.no_grad():
                output = model(points, t, style_features)
            
            # 显示内存使用
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Batch size {bs}: {memory_used:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
            
        except RuntimeError as e:
            print(f"Batch size {bs}: OOM - {e}")
            break


def benchmark_inference_speed():
    """测试推理速度"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = PointCloudDiffusionModel().to(device)
    model.eval()
    
    # 测试参数
    batch_size = 1
    chunk_size = 2048
    num_runs = 100
    
    # 预热
    for _ in range(10):
        points = torch.randn(batch_size, chunk_size, 3).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        style_features = torch.randn(batch_size, 1, 1024).to(device)
        with torch.no_grad():
            _ = model(points, t, style_features)
    
    # 计时
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        points = torch.randn(batch_size, chunk_size, 3).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        style_features = torch.randn(batch_size, 1, 1024).to(device)
        with torch.no_grad():
            _ = model(points, t, style_features)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"\nInference Speed Benchmark:")
    print(f"Average time per batch: {avg_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.2f} batches/second")


if __name__ == "__main__":
    print("Running benchmarks...")
    
    # 内存基准测试
    print("\n=== Memory Benchmark ===")
    benchmark_model_memory()
    
    # 速度基准测试
    print("\n=== Speed Benchmark ===")
    benchmark_inference_speed()
