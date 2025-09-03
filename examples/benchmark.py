"""
性能基准测试脚本 - 分层架构版本
"""

import torch
import time
import numpy as np
from memory_profiler import profile

from models.diffusion_model import PointCloudDiffusionModel
from models.pointnet2_encoder import PointNet2Encoder
from config.config import Config


@profile
def benchmark_model_memory():
    """测试模型内存使用 - 分层架构"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    # 创建模型
    model = PointCloudDiffusionModel(config).to(device)
    encoder = PointNet2Encoder().to(device)
    
    # 测试不同批大小和点云大小
    batch_sizes = [1, 2, 4, 8]
    point_sizes = [30000, 60000, 120000]  # 分层测试：global, medium, full
    
    print("=== Memory Usage Test (Hierarchical Architecture) ===")
    
    for bs in batch_sizes:
        for points_num in point_sizes:
            try:
                # 创建输入 - 分层数据
                points = torch.randn(bs, points_num, 3).to(device)
                condition_points = torch.randn(bs, points_num, 3).to(device)
                t = torch.randint(0, 1000, (bs,)).to(device)
                
                # 前向传播
                with torch.no_grad():
                    # 根据点数自动选择分层策略
                    use_hierarchical = points_num > config.global_points
                    output = model(points, t, condition_points, use_hierarchical=use_hierarchical)
                
                # 显示内存使用
                if device.type == 'cuda':
                    memory_used = torch.cuda.max_memory_allocated() / 1024**3
                    strategy = "Hierarchical" if use_hierarchical else "Direct"
                    print(f"Batch {bs}, Points {points_num//1000}K ({strategy}): {memory_used:.2f} GB")
                    torch.cuda.reset_peak_memory_stats()
                
            except RuntimeError as e:
                print(f"Batch {bs}, Points {points_num//1000}K: OOM - {str(e)[:100]}...")
                break


def benchmark_inference_speed():
    """测试推理速度 - 分层架构优化"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    # 创建模型
    model = PointCloudDiffusionModel(config).to(device)
    model.eval()
    
    # 测试不同规模的点云
    test_configs = [
        {"name": "Global (30K)", "points": 30000, "hierarchical": False},
        {"name": "Medium (60K)", "points": 60000, "hierarchical": True},
        {"name": "Full (120K)", "points": 120000, "hierarchical": True},
    ]
    
    batch_size = 1
    num_runs = 50
    
    print("\n=== Inference Speed Test (Hierarchical Architecture) ===")
    
    for test_config in test_configs:
        points_num = test_config["points"]
        use_hierarchical = test_config["hierarchical"]
        
        print(f"\nTesting {test_config['name']} points...")
        
        # 预热
        for _ in range(5):
            points = torch.randn(batch_size, points_num, 3).to(device)
            condition_points = torch.randn(batch_size, points_num, 3).to(device)
            t = torch.randint(0, 1000, (batch_size,)).to(device)
            with torch.no_grad():
                _ = model(points, t, condition_points, use_hierarchical=use_hierarchical)
        
        # 计时测试
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_runs):
            points = torch.randn(batch_size, points_num, 3).to(device)
            condition_points = torch.randn(batch_size, points_num, 3).to(device)
            t = torch.randint(0, 1000, (batch_size,)).to(device)
            with torch.no_grad():
                _ = model(points, t, condition_points, use_hierarchical=use_hierarchical)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        strategy = "Hierarchical" if use_hierarchical else "Direct"
        
        print(f"  {strategy} processing:")
        print(f"    Average time: {avg_time*1000:.2f} ms")
        print(f"    Throughput: {1/avg_time:.2f} samples/second")
        
        # 计算相对性能
        if test_config["name"] == "Global (30K)":
            baseline_time = avg_time
        else:
            speedup = baseline_time / avg_time if avg_time > 0 else float('inf')
            print(f"    Relative to 30K: {speedup:.2f}x slower")


def benchmark_hierarchical_efficiency():
    """测试分层架构的效率优势"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    model = PointCloudDiffusionModel(config).to(device)
    model.eval()
    
    batch_size = 2
    points_num = 120000  # 完整点云
    num_runs = 20
    
    print("\n=== Hierarchical Efficiency Test ===")
    
    # 测试分层 vs 直接处理的性能差异
    strategies = [
        {"name": "Direct Processing (120K)", "hierarchical": False},
        {"name": "Hierarchical Processing (30K→120K)", "hierarchical": True}
    ]
    
    for strategy in strategies:
        use_hierarchical = strategy["hierarchical"]
        
        # 预热
        for _ in range(3):
            points = torch.randn(batch_size, points_num, 3).to(device)
            condition_points = torch.randn(batch_size, points_num, 3).to(device)
            t = torch.randint(0, 1000, (batch_size,)).to(device)
            
            try:
                with torch.no_grad():
                    _ = model(points, t, condition_points, use_hierarchical=use_hierarchical)
            except RuntimeError as e:
                print(f"Strategy '{strategy['name']}' failed: OOM")
                continue
        
        # 计时测试
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            points = torch.randn(batch_size, points_num, 3).to(device)
            condition_points = torch.randn(batch_size, points_num, 3).to(device)
            t = torch.randint(0, 1000, (batch_size,)).to(device)
            
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    _ = model(points, t, condition_points, use_hierarchical=use_hierarchical)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                if device.type == 'cuda':
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)
                
            except RuntimeError as e:
                print(f"  {strategy['name']}: Failed with OOM")
                break
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_memory = np.mean(memory_usage) if memory_usage else 0
            
            print(f"\n{strategy['name']}:")
            print(f"  Time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
            if avg_memory > 0:
                print(f"  Memory: {avg_memory:.2f} GB")
            print(f"  Success rate: {len(times)}/{num_runs}")


def benchmark_scaling_analysis():
    """分析分层架构的扩展性"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    model = PointCloudDiffusionModel(config).to(device) 
    model.eval()
    
    print("\n=== Scaling Analysis ===")
    
    # 测试不同点云规模
    point_scales = [10000, 30000, 60000, 90000, 120000]
    batch_size = 1
    
    results = []
    
    for points_num in point_scales:
        use_hierarchical = points_num > config.global_points
        
        try:
            # 单次测试
            points = torch.randn(batch_size, points_num, 3).to(device)
            condition_points = torch.randn(batch_size, points_num, 3).to(device)
            t = torch.randint(0, 1000, (batch_size,)).to(device)
            
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(points, t, condition_points, use_hierarchical=use_hierarchical)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            processing_time = end_time - start_time
            memory_used = torch.cuda.max_memory_allocated() / 1024**3 if device.type == 'cuda' else 0
            
            strategy = "Hierarchical" if use_hierarchical else "Direct"
            
            results.append({
                'points': points_num,
                'time': processing_time,
                'memory': memory_used,
                'strategy': strategy
            })
            
            print(f"{points_num//1000}K points ({strategy}): {processing_time*1000:.2f}ms, {memory_used:.2f}GB")
            
        except RuntimeError as e:
            print(f"{points_num//1000}K points: OOM")
    
    # 分析结果
    if len(results) > 1:
        print("\nScaling Analysis:")
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            
            time_ratio = curr['time'] / prev['time']
            memory_ratio = curr['memory'] / prev['memory'] if prev['memory'] > 0 else 1
            point_ratio = curr['points'] / prev['points']
            
            print(f"  {prev['points']//1000}K → {curr['points']//1000}K:")
            print(f"    Points: {point_ratio:.2f}x, Time: {time_ratio:.2f}x, Memory: {memory_ratio:.2f}x")


if __name__ == "__main__":
    print("Running Point Cloud Style Transfer Benchmarks...")
    print("Architecture: Hierarchical (Global + Local)")
    
    # 内存基准测试
    print("\n" + "="*60)
    benchmark_model_memory()
    
    # 速度基准测试
    print("\n" + "="*60)
    benchmark_inference_speed()
    
    # 分层效率测试
    print("\n" + "="*60)
    benchmark_hierarchical_efficiency()
    
    # 扩展性分析
    print("\n" + "="*60)
    benchmark_scaling_analysis()
    
    print("\n" + "="*60)
    print("Benchmark completed!")