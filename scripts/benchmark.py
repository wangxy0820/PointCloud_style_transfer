#!/usr/bin/env python3
"""
基准测试脚本
用于评估模型性能、训练速度和推理效率
"""

import argparse
import os
import sys
import torch
import numpy as np
import time
import json
from datetime import datetime
import psutil
import GPUtil
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models.generator import CycleConsistentGenerator, PointCloudGenerator
from models.discriminator import HybridDiscriminator, PointCloudDiscriminator
from models.pointnet2 import PointNet2AutoEncoder
from evaluation.metrics import PointCloudMetrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Benchmark Point Cloud Models')
    
    # 基准测试类型
    parser.add_argument('--benchmark_type', type=str, default='all',
                       choices=['model_speed', 'memory_usage', 'accuracy', 'scalability', 'all'],
                       help='Type of benchmark to run')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default='',
                       help='Path to trained model (for accuracy benchmark)')
    parser.add_argument('--chunk_size', type=int, default=8192,
                       help='Point cloud chunk size')
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Latent dimension size')
    parser.add_argument('--generator_dim', type=int, default=256,
                       help='Generator style dimension')
    
    # 测试参数
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--point_counts', type=int, nargs='+', 
                       default=[1024, 2048, 4096, 8192, 16384],
                       help='Point counts to test')
    parser.add_argument('--num_iterations', type=int, default=100,
                       help='Number of iterations for speed test')
    parser.add_argument('--warmup_iterations', type=int, default=10,
                       help='Number of warmup iterations')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'both'],
                       help='Device to test on')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                       help='Directory to save benchmark results')
    parser.add_argument('--save_detailed_logs', action='store_true',
                       help='Save detailed per-iteration logs')
    
    return parser.parse_args()


class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.cpu_percent = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
    
    def start_monitoring(self):
        """开始监控"""
        self.cpu_percent = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
    
    def record_metrics(self):
        """记录当前指标"""
        # CPU和内存使用率
        self.cpu_percent.append(psutil.cpu_percent())
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.used / (1024**3))  # GB
        
        # GPU使用率
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 使用第一个GPU
                self.gpu_usage.append(gpu.load * 100)
                self.gpu_memory.append(gpu.memoryUsed / 1024)  # GB
            else:
                self.gpu_usage.append(0)
                self.gpu_memory.append(0)
        except:
            self.gpu_usage.append(0)
            self.gpu_memory.append(0)
    
    def get_summary(self):
        """获取监控摘要"""
        if not self.cpu_percent:
            return {}
        
        return {
            'cpu_percent': {
                'mean': np.mean(self.cpu_percent),
                'max': np.max(self.cpu_percent),
                'std': np.std(self.cpu_percent)
            },
            'memory_usage_gb': {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'std': np.std(self.memory_usage)
            },
            'gpu_usage_percent': {
                'mean': np.mean(self.gpu_usage),
                'max': np.max(self.gpu_usage),
                'std': np.std(self.gpu_usage)
            },
            'gpu_memory_gb': {
                'mean': np.mean(self.gpu_memory),
                'max': np.max(self.gpu_memory),
                'std': np.std(self.gpu_memory)
            }
        }


class ModelBenchmark:
    """模型基准测试器"""
    
    def __init__(self, args):
        self.args = args
        self.monitor = SystemMonitor()
        self.results = {}
        
        # 设置设备
        if args.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.gpu_id}')
            torch.cuda.set_device(args.gpu_id)
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
    
    def create_models(self, num_points: int):
        """创建测试模型"""
        config = Config()
        config.chunk_size = num_points
        config.latent_dim = self.args.latent_dim
        config.generator_dim = self.args.generator_dim
        
        models = {}
        
        # PointNet++自编码器
        models['pointnet2_ae'] = PointNet2AutoEncoder(
            input_channels=3,
            latent_dim=config.latent_dim,
            num_points=num_points
        ).to(self.device)
        
        # 点云生成器
        models['generator'] = PointCloudGenerator(
            input_channels=3,
            style_dim=config.generator_dim,
            latent_dim=config.latent_dim,
            num_points=num_points
        ).to(self.device)
        
        # 循环一致性生成器
        models['cycle_generator'] = CycleConsistentGenerator(
            input_channels=3,
            style_dim=config.generator_dim,
            latent_dim=config.latent_dim,
            num_points=num_points
        ).to(self.device)
        
        # 判别器
        models['discriminator'] = PointCloudDiscriminator(
            input_channels=3
        ).to(self.device)
        
        # 混合判别器
        models['hybrid_discriminator'] = HybridDiscriminator(
            input_channels=3,
            patch_size=min(1024, num_points // 8)
        ).to(self.device)
        
        return models
    
    def create_test_data(self, batch_size: int, num_points: int):
        """创建测试数据"""
        # 生成随机点云数据
        data = {
            'sim_points': torch.randn(batch_size, num_points, 3).to(self.device),
            'real_points': torch.randn(batch_size, num_points, 3).to(self.device),
            'input_points': torch.randn(batch_size, num_points, 3).to(self.device)
        }
        return data
    
    def benchmark_model_speed(self):
        """基准测试模型速度"""
        print("Running model speed benchmark...")
        
        speed_results = {}
        
        for num_points in self.args.point_counts:
            print(f"\nTesting with {num_points} points...")
            
            models = self.create_models(num_points)
            speed_results[num_points] = {}
            
            for batch_size in self.args.batch_sizes:
                print(f"  Batch size: {batch_size}")
                
                # 创建测试数据
                test_data = self.create_test_data(batch_size, num_points)
                
                batch_results = {}
                
                for model_name, model in models.items():
                    model.eval()
                    
                    # 预热
                    with torch.no_grad():
                        for _ in range(self.args.warmup_iterations):
                            try:
                                if model_name == 'pointnet2_ae':
                                    _ = model(test_data['input_points'])
                                elif model_name == 'generator':
                                    _ = model(test_data['sim_points'], test_data['real_points'])
                                elif model_name == 'cycle_generator':
                                    _ = model(test_data['sim_points'], test_data['real_points'])
                                elif 'discriminator' in model_name:
                                    _ = model(test_data['input_points'])
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    print(f"    {model_name}: OOM")
                                    batch_results[model_name] = {'status': 'OOM'}
                                    break
                                else:
                                    raise e
                    
                    if model_name in batch_results and batch_results[model_name]['status'] == 'OOM':
                        continue
                    
                    # 计时测试
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(self.args.num_iterations):
                            try:
                                if model_name == 'pointnet2_ae':
                                    _ = model(test_data['input_points'])
                                elif model_name == 'generator':
                                    _ = model(test_data['sim_points'], test_data['real_points'])
                                elif model_name == 'cycle_generator':
                                    _ = model(test_data['sim_points'], test_data['real_points'])
                                elif 'discriminator' in model_name:
                                    _ = model(test_data['input_points'])
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    batch_results[model_name] = {'status': 'OOM'}
                                    break
                                else:
                                    raise e
                    
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end_time = time.time()
                    
                    if model_name not in batch_results:
                        total_time = end_time - start_time
                        avg_time = total_time / self.args.num_iterations
                        throughput = batch_size / avg_time  # samples per second
                        
                        batch_results[model_name] = {
                            'status': 'success',
                            'avg_time_per_batch': avg_time,
                            'throughput_samples_per_sec': throughput,
                            'total_time': total_time
                        }
                    
                    # 清理GPU内存
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                speed_results[num_points][batch_size] = batch_results
        
        self.results['speed'] = speed_results
    
    def benchmark_memory_usage(self):
        """基准测试内存使用"""
        print("Running memory usage benchmark...")
        
        memory_results = {}
        
        for num_points in self.args.point_counts:
            print(f"\nTesting memory with {num_points} points...")
            
            models = self.create_models(num_points)
            memory_results[num_points] = {}
            
            for batch_size in self.args.batch_sizes:
                print(f"  Batch size: {batch_size}")
                
                # 清理内存
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                test_data = self.create_test_data(batch_size, num_points)
                batch_results = {}
                
                for model_name, model in models.items():
                    model.eval()
                    
                    # 记录初始内存
                    if self.device.type == 'cuda':
                        initial_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                    
                    try:
                        with torch.no_grad():
                            if model_name == 'pointnet2_ae':
                                output = model(test_data['input_points'])
                            elif model_name == 'generator':
                                output = model(test_data['sim_points'], test_data['real_points'])
                            elif model_name == 'cycle_generator':
                                output = model(test_data['sim_points'], test_data['real_points'])
                            elif 'discriminator' in model_name:
                                output = model(test_data['input_points'])
                        
                        # 记录峰值内存
                        if self.device.type == 'cuda':
                            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                            memory_used = peak_memory - initial_memory
                        else:
                            memory_used = 0  # CPU内存测量较复杂，暂时设为0
                        
                        batch_results[model_name] = {
                            'status': 'success',
                            'memory_used_gb': memory_used,
                            'peak_memory_gb': peak_memory if self.device.type == 'cuda' else 0
                        }
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            batch_results[model_name] = {'status': 'OOM'}
                        else:
                            raise e
                    
                    # 清理内存
                    if 'output' in locals():
                        del output
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                memory_results[num_points][batch_size] = batch_results
        
        self.results['memory'] = memory_results
    
    def benchmark_scalability(self):
        """基准测试可扩展性"""
        print("Running scalability benchmark...")
        
        scalability_results = {}
        
        # 测试不同规模的数据
        large_point_counts = [8192, 16384, 32768, 65536, 131072]
        test_batch_size = 1  # 使用小批次大小测试大规模数据
        
        for num_points in large_point_counts:
            if num_points > max(self.args.point_counts):
                print(f"\nTesting scalability with {num_points} points...")
                
                try:
                    # 只测试核心模型
                    model = PointNet2AutoEncoder(
                        input_channels=3,
                        latent_dim=512,
                        num_points=min(num_points, 8192)  # 限制模型大小
                    ).to(self.device)
                    
                    # 分块处理大规模数据
                    chunk_size = 8192
                    num_chunks = (num_points + chunk_size - 1) // chunk_size
                    
                    test_data = torch.randn(test_batch_size, num_points, 3).to(self.device)
                    
                    start_time = time.time()
                    
                    with torch.no_grad():
                        chunks = []
                        for i in range(num_chunks):
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, num_points)
                            chunk = test_data[:, start_idx:end_idx, :]
                            
                            if chunk.size(1) < chunk_size:
                                # 填充到标准大小
                                padding = torch.zeros(test_batch_size, chunk_size - chunk.size(1), 3).to(self.device)
                                chunk = torch.cat([chunk, padding], dim=1)
                            
                            chunk_output = model(chunk)
                            chunks.append(chunk_output[0][:, :end_idx-start_idx, :])  # 移除填充
                        
                        # 合并结果
                        result = torch.cat(chunks, dim=1)
                    
                    end_time = time.time()
                    
                    scalability_results[num_points] = {
                        'status': 'success',
                        'processing_time': end_time - start_time,
                        'num_chunks': num_chunks,
                        'throughput_points_per_sec': num_points / (end_time - start_time)
                    }
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        scalability_results[num_points] = {'status': 'OOM'}
                    else:
                        raise e
                
                # 清理内存
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        self.results['scalability'] = scalability_results
    
    def save_results(self):
        """保存基准测试结果"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # 添加系统信息
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if torch.cuda.is_available():
            gpu_info = {
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
            system_info.update(gpu_info)
        
        # 保存完整结果
        full_results = {
            'system_info': system_info,
            'benchmark_config': vars(self.args),
            'results': self.results
        }
        
        with open(os.path.join(self.args.output_dir, 'benchmark_results.json'), 'w') as f:
            json.dump(full_results, f, indent=2)
        
        # 生成摘要报告
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """生成摘要报告"""
        summary = {
            'benchmark_summary': {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'test_configurations': {
                    'point_counts': self.args.point_counts,
                    'batch_sizes': self.args.batch_sizes,
                    'iterations': self.args.num_iterations
                }
            }
        }
        
        # 速度基准摘要
        if 'speed' in self.results:
            speed_summary = {}
            for num_points, batch_data in self.results['speed'].items():
                for batch_size, model_data in batch_data.items():
                    key = f"{num_points}pts_bs{batch_size}"
                    speed_summary[key] = {
                        model: data.get('throughput_samples_per_sec', 0) if data.get('status') == 'success' else 'OOM'
                        for model, data in model_data.items()
                    }
            summary['speed_summary'] = speed_summary
        
        # 内存使用摘要
        if 'memory' in self.results:
            memory_summary = {}
            for num_points, batch_data in self.results['memory'].items():
                for batch_size, model_data in batch_data.items():
                    key = f"{num_points}pts_bs{batch_size}"
                    memory_summary[key] = {
                        model: data.get('memory_used_gb', 0) if data.get('status') == 'success' else 'OOM'
                        for model, data in model_data.items()
                    }
            summary['memory_summary'] = memory_summary
        
        # 可扩展性摘要
        if 'scalability' in self.results:
            summary['scalability_summary'] = self.results['scalability']
        
        # 保存摘要
        with open(os.path.join(self.args.output_dir, 'benchmark_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 打印摘要
        self.print_summary(summary)
    
    def print_summary(self, summary):
        """打印基准测试摘要"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        print(f"Device: {summary['benchmark_summary']['device']}")
        print(f"Date: {summary['benchmark_summary']['date']}")
        
        if 'speed_summary' in summary:
            print(f"\nSPEED BENCHMARK (samples/sec):")
            print("-" * 40)
            for config, models in list(summary['speed_summary'].items())[:5]:  # 只显示前5个配置
                print(f"\n{config}:")
                for model, speed in models.items():
                    if speed == 'OOM':
                        print(f"  {model}: Out of Memory")
                    else:
                        print(f"  {model}: {speed:.2f}")
        
        if 'memory_summary' in summary:
            print(f"\nMEMORY USAGE (GB):")
            print("-" * 40)
            for config, models in list(summary['memory_summary'].items())[:3]:  # 只显示前3个配置
                print(f"\n{config}:")
                for model, memory in models.items():
                    if memory == 'OOM':
                        print(f"  {model}: Out of Memory")
                    else:
                        print(f"  {model}: {memory:.2f} GB")
        
        print(f"\nFull results saved to: {self.args.output_dir}")
        print("="*80)
    
    def run_benchmark(self):
        """运行基准测试"""
        print(f"Starting benchmark on {self.device}")
        print(f"Test configurations: {len(self.args.point_counts)} point counts, {len(self.args.batch_sizes)} batch sizes")
        
        if self.args.benchmark_type in ['model_speed', 'all']:
            self.benchmark_model_speed()
        
        if self.args.benchmark_type in ['memory_usage', 'all']:
            self.benchmark_memory_usage()
        
        if self.args.benchmark_type in ['scalability', 'all']:
            self.benchmark_scalability()
        
        self.save_results()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建基准测试器
    benchmark = ModelBenchmark(args)
    
    # 运行基准测试
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()