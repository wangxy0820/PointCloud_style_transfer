#!/usr/bin/env python3
"""
优化的训练启动脚本，解决内存碎片化问题
"""

import os
import sys
import torch
import gc
import subprocess

def cleanup_gpu():
    """彻底清理GPU内存"""
    if torch.cuda.is_available():
        # 清空所有CUDA缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 重置内存分配器
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # 强制垃圾回收
        gc.collect()
        
        print("GPU memory cleaned")
        
        # 显示当前内存状态
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def kill_other_gpu_processes():
    """杀死其他占用GPU的Python进程"""
    try:
        # 获取当前进程ID
        current_pid = os.getpid()
        
        # 查找占用GPU的进程
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
            
            for pid in pids:
                if pid != current_pid:
                    try:
                        os.kill(pid, 9)
                        print(f"Killed process {pid}")
                    except:
                        pass
    except:
        print("Could not check for other GPU processes")

def set_memory_optimizations():
    """设置内存优化"""
    # 环境变量设置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:64'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步执行，减少内存峰值
    
    if torch.cuda.is_available():
        # 设置内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.8)  # 只使用80%的GPU内存
        
        # 禁用cudnn benchmark（可能会增加内存使用）
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_optimal_config():
    """根据GPU内存自动确定最优配置"""
    if not torch.cuda.is_available():
        return {
            'batch_size': 1,
            'chunk_size': 1024,
            'channels': [32, 64, 128, 256],
            'latent_dim': 256,
            'generator_dim': 128
        }
    
    # 获取GPU内存
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_memory_gb >= 24:  # 24GB或更多
        return {
            'batch_size': 4,
            'chunk_size': 4096,
            'channels': [64, 128, 256, 512],
            'latent_dim': 512,
            'generator_dim': 256
        }
    elif gpu_memory_gb >= 16:  # 16-24GB
        return {
            'batch_size': 2,
            'chunk_size': 4096,
            'channels': [32, 64, 128, 256],
            'latent_dim': 256,
            'generator_dim': 128
        }
    elif gpu_memory_gb >= 12:  # 12-16GB (你的情况)
        return {
            'batch_size': 2,
            'chunk_size': 2048,
            'channels': [32, 64, 128, 256],
            'latent_dim': 256,
            'generator_dim': 128
        }
    else:  # 小于12GB
        return {
            'batch_size': 1,
            'chunk_size': 1024,
            'channels': [32, 64, 128, 256],
            'latent_dim': 128,
            'generator_dim': 64
        }

def main():
    """主函数"""
    print("=== Optimized Training Launcher ===\n")
    
    # 1. 清理环境
    print("Step 1: Cleaning environment...")
    kill_other_gpu_processes()
    cleanup_gpu()
    
    # 2. 设置优化
    print("\nStep 2: Setting memory optimizations...")
    set_memory_optimizations()
    
    # 3. 获取最优配置
    print("\nStep 3: Determining optimal configuration...")
    config = get_optimal_config()
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    print(f"Recommended configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 4. 替换pointnet2.py文件
    print("\nStep 4: Updating model files...")
    if os.path.exists('pointnet2_complete_fix.py'):
        import shutil
        shutil.copy('pointnet2_complete_fix.py', 'models/pointnet2.py')
        print("✓ Updated pointnet2.py with fixed version")
    
    # 5. 构建训练命令
    print("\nStep 5: Starting training...")
    
    # 基础命令
    cmd = [
        'python', 'scripts/train.py',
        '--data_dir', 'datasets/processed',
        '--experiment_name', 'optimized_experiment',
        '--batch_size', str(config['batch_size']),
        '--chunk_size', str(config['chunk_size']),
        '--latent_dim', str(config['latent_dim']),
        '--generator_dim', str(config['generator_dim']),
        '--num_workers', '2',  # 减少数据加载进程
        '--memory_efficient',
        '--save_interval', '20',
        '--eval_interval', '10',
        '--log_interval', '50'
    ]
    
    # 询问是否使用推荐配置
    response = input("\nUse recommended configuration? (y/n): ").strip().lower()
    if response == 'n':
        # 手动输入配置
        batch_size = input(f"Batch size (default {config['batch_size']}): ").strip()
        chunk_size = input(f"Chunk size (default {config['chunk_size']}): ").strip()
        
        if batch_size:
            cmd[cmd.index('--batch_size') + 1] = batch_size
        if chunk_size:
            cmd[cmd.index('--chunk_size') + 1] = chunk_size
    
    # 执行训练
    print("\nExecuting command:")
    print(' '.join(cmd))
    print("\n" + "="*50 + "\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        
        # 如果失败，尝试更小的配置
        print("\nTrying with minimal configuration...")
        cmd_minimal = [
            'python', 'scripts/train.py',
            '--data_dir', 'datasets/processed',
            '--experiment_name', 'minimal_experiment',
            '--batch_size', '1',
            '--chunk_size', '1024',
            '--latent_dim', '128',
            '--generator_dim', '64',
            '--num_workers', '1',
            '--memory_efficient'
        ]
        subprocess.run(cmd_minimal)

if __name__ == "__main__":
    main()