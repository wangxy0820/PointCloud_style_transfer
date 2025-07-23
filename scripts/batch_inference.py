#!/usr/bin/env python3
"""
批量推理脚本
用于批量处理多个点云文件的风格转换
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm
import time
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference import DiffusionInference


def process_single_file(args):
    """处理单个文件的函数（用于多进程）"""
    sim_file, real_reference_path, output_path, checkpoint_path, device_id = args
    
    try:
        # 设置设备
        device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        
        # 创建推理器
        inference = DiffusionInference(checkpoint_path, device)
        
        # 加载数据
        sim_points = np.load(sim_file)
        real_reference = np.load(real_reference_path)
        
        # 风格转换
        start_time = time.time()
        transferred = inference.transfer_style(sim_points, real_reference)
        inference_time = time.time() - start_time
        
        # 保存结果
        np.save(output_path, transferred.astype(np.float32))
        
        return {
            'status': 'success',
            'input_file': sim_file,
            'output_file': output_path,
            'inference_time': inference_time,
            'input_points': len(sim_points),
            'output_points': len(transferred)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'input_file': sim_file,
            'error': str(e)
        }


class BatchInference:
    """批量推理管理器"""
    
    def __init__(self, checkpoint_path: str, num_gpus: int = 1, batch_size: int = 1):
        self.checkpoint_path = checkpoint_path
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        
    def process_folder(self, 
                      sim_folder: str,
                      real_reference: str,
                      output_folder: str,
                      pattern: str = "*.npy",
                      max_files: int = 0,
                      num_workers: int = 1):
        """
        批量处理文件夹
        
        Args:
            sim_folder: 仿真点云文件夹
            real_reference: 真实参考点云路径（单个文件或文件夹）
            output_folder: 输出文件夹
            pattern: 文件匹配模式
            max_files: 最大处理文件数（0表示全部）
            num_workers: 并行工作进程数
        """
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取输入文件列表
        sim_files = sorted(glob.glob(os.path.join(sim_folder, pattern)))
        if not sim_files:
            print(f"No files found matching pattern '{pattern}' in {sim_folder}")
            return
        
        if max_files > 0:
            sim_files = sim_files[:max_files]
        
        print(f"Found {len(sim_files)} simulation files to process")
        
        # 处理真实参考
        if os.path.isfile(real_reference):
            # 单个参考文件
            real_references = [real_reference] * len(sim_files)
        else:
            # 文件夹，为每个仿真文件匹配一个真实文件
            real_files = sorted(glob.glob(os.path.join(real_reference, pattern)))
            if not real_files:
                print(f"No reference files found in {real_reference}")
                return
            
            # 循环使用真实文件
            real_references = []
            for i in range(len(sim_files)):
                real_idx = i % len(real_files)
                real_references.append(real_files[real_idx])
        
        # 准备任务列表
        tasks = []
        for i, (sim_file, real_ref) in enumerate(zip(sim_files, real_references)):
            filename = os.path.basename(sim_file)
            output_path = os.path.join(output_folder, filename.replace('.npy', '_transferred.npy'))
            
            # 分配GPU（循环使用）
            gpu_id = i % self.num_gpus
            
            tasks.append((sim_file, real_ref, output_path, self.checkpoint_path, gpu_id))
        
        # 开始处理
        start_time = time.time()
        results = []
        
        if num_workers > 1:
            # 多进程处理
            print(f"Processing with {num_workers} workers...")
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 提交任务
                future_to_task = {executor.submit(process_single_file, task): task for task in tasks}
                
                # 处理完成的任务
                with tqdm(total=len(tasks), desc="Processing files") as pbar:
                    for future in as_completed(future_to_task):
                        result = future.result()
                        results.append(result)
                        
                        if result['status'] == 'success':
                            pbar.set_postfix({
                                'file': os.path.basename(result['input_file']),
                                'time': f"{result['inference_time']:.2f}s"
                            })
                        else:
                            print(f"\nError processing {result['input_file']}: {result.get('error', 'Unknown error')}")
                        
                        pbar.update(1)
        else:
            # 单进程处理
            print("Processing with single worker...")
            
            for task in tqdm(tasks, desc="Processing files"):
                result = process_single_file(task)
                results.append(result)
                
                if result['status'] == 'error':
                    print(f"\nError processing {result['input_file']}: {result.get('error', 'Unknown error')}")
        
        # 统计结果
        total_time = time.time() - start_time
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        # 保存处理报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': self.checkpoint_path,
            'total_files': len(tasks),
            'successful': len(successful),
            'failed': len(failed),
            'total_time': total_time,
            'average_time': total_time / len(successful) if successful else 0,
            'results': results
        }
        
        report_path = os.path.join(output_folder, 'batch_inference_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印摘要
        print(f"\n{'='*60}")
        print("BATCH INFERENCE COMPLETED")
        print(f"{'='*60}")
        print(f"Total files: {len(tasks)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per file: {total_time/len(successful):.2f} seconds" if successful else "N/A")
        print(f"Output directory: {output_folder}")
        print(f"Report saved to: {report_path}")
        
        if failed:
            print(f"\nFailed files:")
            for f in failed[:5]:  # 只显示前5个
                print(f"  - {f['input_file']}: {f.get('error', 'Unknown error')}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")
    
    def process_with_different_references(self,
                                        sim_file: str,
                                        real_references: list,
                                        output_folder: str):
        """
        使用不同的参考点云处理同一个仿真点云
        
        Args:
            sim_file: 仿真点云文件
            real_references: 真实参考点云列表
            output_folder: 输出文件夹
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # 创建推理器
        inference = DiffusionInference(self.checkpoint_path)
        
        # 加载仿真点云
        sim_points = np.load(sim_file)
        sim_basename = os.path.basename(sim_file).replace('.npy', '')
        
        print(f"Processing {sim_basename} with {len(real_references)} different references...")
        
        for i, real_ref_path in enumerate(tqdm(real_references, desc="Processing references")):
            # 加载参考
            real_reference = np.load(real_ref_path)
            ref_basename = os.path.basename(real_ref_path).replace('.npy', '')
            
            # 风格转换
            transferred = inference.transfer_style(sim_points, real_reference)
            
            # 保存结果
            output_filename = f"{sim_basename}_ref_{ref_basename}.npy"
            output_path = os.path.join(output_folder, output_filename)
            np.save(output_path, transferred.astype(np.float32))
        
        print(f"Completed! Results saved to {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Batch inference for point cloud style transfer')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--sim_dir', type=str, required=True,
                       help='Directory containing simulation point clouds')
    parser.add_argument('--real_reference', type=str, required=True,
                       help='Real reference point cloud file or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    
    # 可选参数
    parser.add_argument('--pattern', type=str, default='*.npy',
                       help='File pattern to match')
    parser.add_argument('--max_files', type=int, default=0,
                       help='Maximum number of files to process (0 = all)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use')
    
    # 特殊模式
    parser.add_argument('--multi_ref_mode', action='store_true',
                       help='Process single sim file with multiple references')
    parser.add_argument('--sim_file', type=str, default='',
                       help='Single simulation file (for multi_ref_mode)')
    
    args = parser.parse_args()
    
    # 创建批量推理器
    batch_inference = BatchInference(
        args.checkpoint,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size
    )
    
    if args.multi_ref_mode:
        # 多参考模式
        if not args.sim_file or not os.path.isfile(args.sim_file):
            print("Error: --sim_file must be specified in multi_ref_mode")
            sys.exit(1)
        
        # 获取所有参考文件
        if os.path.isdir(args.real_reference):
            real_references = sorted(glob.glob(os.path.join(args.real_reference, args.pattern)))
        else:
            real_references = [args.real_reference]
        
        batch_inference.process_with_different_references(
            args.sim_file,
            real_references,
            args.output_dir
        )
    else:
        # 标准批量处理模式
        batch_inference.process_folder(
            args.sim_dir,
            args.real_reference,
            args.output_dir,
            pattern=args.pattern,
            max_files=args.max_files,
            num_workers=args.num_workers
        )


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()
