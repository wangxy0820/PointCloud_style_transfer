import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import json
import time
from datetime import datetime

from .metrics import PointCloudMetrics, ClassificationMetrics
from visualization.visualize import PointCloudVisualizer


class StyleTransferEvaluator:
    """风格迁移模型评估器"""
    
    def __init__(self, device: str = 'cuda', output_dir: str = 'evaluation_results'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.metrics_calculator = PointCloudMetrics(device=device)
        self.visualizer = PointCloudVisualizer()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_generation_quality(self, generated: torch.Tensor, 
                                  reference: torch.Tensor,
                                  detailed: bool = True) -> Dict[str, float]:
        """
        评估生成质量
        Args:
            generated: 生成的点云 [B, N, 3]
            reference: 参考点云 [B, M, 3]
            detailed: 是否计算详细指标
        Returns:
            评估指标字典
        """
        results = {}
        
        # 基础几何指标
        cd = self.metrics_calculator.chamfer_distance(generated, reference)
        results['chamfer_distance'] = cd.mean().item()
        results['chamfer_distance_std'] = cd.std().item()
        
        hd = self.metrics_calculator.hausdorff_distance(generated, reference)
        results['hausdorff_distance'] = hd.mean().item()
        results['hausdorff_distance_std'] = hd.std().item()
        
        # EMD（如果点数相同）
        if generated.size(1) == reference.size(1):
            emd = self.metrics_calculator.earth_mover_distance(generated, reference)
            results['earth_mover_distance'] = emd.mean().item()
            results['earth_mover_distance_std'] = emd.std().item()
        
        if detailed:
            # 覆盖度评分
            coverage = self.metrics_calculator.coverage_score(generated, reference)
            results['coverage_score'] = coverage
            
            # 均匀性评分
            uniformity = self.metrics_calculator.uniformity_score(generated)
            results['uniformity_score'] = uniformity
            
            # 局部特征统计
            gen_stats = self.metrics_calculator.local_feature_statistics(generated)
            ref_stats = self.metrics_calculator.local_feature_statistics(reference)
            
            for key in gen_stats:
                results[f'generated_{key}'] = gen_stats[key]
                results[f'reference_{key}'] = ref_stats[key]
                
                # 计算差异
                if isinstance(gen_stats[key], (int, float)) and isinstance(ref_stats[key], (int, float)):
                    results[f'{key}_difference'] = abs(gen_stats[key] - ref_stats[key])
        
        return results
    
    def evaluate_content_preservation(self, generated: torch.Tensor,
                                    original: torch.Tensor) -> Dict[str, float]:
        """
        评估内容保持能力
        Args:
            generated: 生成的点云 [B, N, 3]
            original: 原始点云 [B, N, 3]
        Returns:
            内容保持指标
        """
        results = {}
        
        # 几何保持
        cd = self.metrics_calculator.chamfer_distance(generated, original)
        results['content_chamfer_distance'] = cd.mean().item()
        
        # 结构保持（使用局部密度相似性）
        gen_stats = self.metrics_calculator.local_feature_statistics(generated)
        orig_stats = self.metrics_calculator.local_feature_statistics(original)
        
        # 密度相似性
        density_similarity = 1.0 / (1.0 + abs(gen_stats['mean_local_density'] - orig_stats['mean_local_density']))
        results['density_similarity'] = density_similarity
        
        # 形状保持（使用主成分分析）
        shape_similarity = self._compute_shape_similarity(generated, original)
        results['shape_similarity'] = shape_similarity
        
        return results
    
    def evaluate_style_transfer_effectiveness(self, original: torch.Tensor,
                                            generated: torch.Tensor,
                                            style_reference: torch.Tensor) -> Dict[str, float]:
        """
        评估风格迁移效果
        Args:
            original: 原始点云 [B, N, 3]
            generated: 生成点云 [B, N, 3]
            style_reference: 风格参考点云 [B, M, 3]
        Returns:
            风格迁移效果指标
        """
        results = {}
        
        # 计算到风格参考的距离改善
        orig_to_style_dist = self.metrics_calculator.chamfer_distance(
            original, style_reference, bidirectional=False
        ).mean().item()
        
        gen_to_style_dist = self.metrics_calculator.chamfer_distance(
            generated, style_reference, bidirectional=False
        ).mean().item()
        
        # 风格迁移比率
        if orig_to_style_dist > 0:
            style_transfer_ratio = max(0, 1 - (gen_to_style_dist / orig_to_style_dist))
        else:
            style_transfer_ratio = 0.0
        
        results['style_transfer_ratio'] = style_transfer_ratio
        results['original_to_style_distance'] = orig_to_style_dist
        results['generated_to_style_distance'] = gen_to_style_dist
        
        # 风格特征相似性
        style_similarity = self._compute_style_similarity(generated, style_reference)
        results['style_feature_similarity'] = style_similarity
        
        return results
    
    def evaluate_cycle_consistency(self, original: torch.Tensor,
                                 cycled: torch.Tensor) -> Dict[str, float]:
        """
        评估循环一致性
        Args:
            original: 原始点云 [B, N, 3]
            cycled: 循环重建的点云 [B, N, 3]
        Returns:
            循环一致性指标
        """
        results = {}
        
        # 循环重建误差
        cd = self.metrics_calculator.chamfer_distance(original, cycled)
        results['cycle_consistency_error'] = cd.mean().item()
        results['cycle_consistency_error_std'] = cd.std().item()
        
        # 点对点误差（如果点数相同）
        if original.size(1) == cycled.size(1):
            point_wise_error = torch.mean(torch.norm(original - cycled, dim=2), dim=1)
            results['cycle_pointwise_error'] = point_wise_error.mean().item()
            results['cycle_pointwise_error_std'] = point_wise_error.std().item()
        
        return results
    
    def evaluate_identity_preservation(self, input_same_domain: torch.Tensor,
                                     identity_output: torch.Tensor) -> Dict[str, float]:
        """
        评估身份保持（同域映射）
        Args:
            input_same_domain: 同域输入 [B, N, 3]
            identity_output: 身份映射输出 [B, N, 3]
        Returns:
            身份保持指标
        """
        results = {}
        
        # 身份映射误差
        cd = self.metrics_calculator.chamfer_distance(input_same_domain, identity_output)
        results['identity_error'] = cd.mean().item()
        results['identity_error_std'] = cd.std().item()
        
        return results
    
    def _compute_shape_similarity(self, pc1: torch.Tensor, pc2: torch.Tensor) -> float:
        """
        计算形状相似性（基于主成分分析）
        Args:
            pc1: 点云1 [B, N, 3]
            pc2: 点云2 [B, N, 3]
        Returns:
            形状相似性分数
        """
        similarities = []
        
        for b in range(pc1.size(0)):
            # 中心化
            points1 = pc1[b] - pc1[b].mean(dim=0, keepdim=True)
            points2 = pc2[b] - pc2[b].mean(dim=0, keepdim=True)
            
            # 计算协方差矩阵
            cov1 = torch.mm(points1.T, points1) / (points1.size(0) - 1)
            cov2 = torch.mm(points2.T, points2) / (points2.size(0) - 1)
            
            # 计算特征值
            try:
                eigenvals1 = torch.linalg.eigvals(cov1).real
                eigenvals2 = torch.linalg.eigvals(cov2).real
                
                # 排序特征值
                eigenvals1 = torch.sort(eigenvals1, descending=True)[0]
                eigenvals2 = torch.sort(eigenvals2, descending=True)[0]
                
                # 计算相似性
                diff = torch.abs(eigenvals1 - eigenvals2)
                similarity = torch.exp(-torch.sum(diff)).item()
                similarities.append(similarity)
                
            except:
                similarities.append(0.5)  # 默认中等相似性
        
        return np.mean(similarities)
    
    def _compute_style_similarity(self, pc1: torch.Tensor, pc2: torch.Tensor) -> float:
        """
        计算风格相似性（基于局部特征分布）
        Args:
            pc1: 点云1 [B, N, 3]
            pc2: 点云2 [B, N, 3]
        Returns:
            风格相似性分数
        """
        # 计算局部特征统计
        stats1 = self.metrics_calculator.local_feature_statistics(pc1)
        stats2 = self.metrics_calculator.local_feature_statistics(pc2)
        
        # 比较关键统计量
        key_stats = ['mean_local_density', 'std_local_density', 'mean_k_distance', 'std_k_distance']
        
        similarities = []
        for stat in key_stats:
            if stat in stats1 and stat in stats2:
                val1, val2 = stats1[stat], stats2[stat]
                if val1 > 0 and val2 > 0:
                    sim = min(val1, val2) / max(val1, val2)
                else:
                    sim = 1.0 if val1 == val2 else 0.0
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def evaluate_model_comprehensive(self, model: nn.Module,
                                   test_data: torch.utils.data.DataLoader,
                                   save_examples: bool = True,
                                   num_examples: int = 5) -> Dict[str, Any]:
        """
        全面评估模型
        Args:
            model: 待评估的模型
            test_data: 测试数据加载器
            save_examples: 是否保存可视化例子
            num_examples: 保存例子的数量
        Returns:
            完整的评估结果
        """
        model.eval()
        
        all_results = {
            'generation_quality': [],
            'content_preservation': [],
            'style_transfer': [],
            'cycle_consistency': [],
            'identity_preservation': [],
            'processing_times': []
        }
        
        example_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_data, desc="Evaluating")):
                sim_points = batch['sim_points'].to(self.device)
                real_points = batch['real_points'].to(self.device)
                
                # 记录处理时间
                start_time = time.time()
                
                # 生成数据
                if hasattr(model, 'sim2real') and hasattr(model, 'real2sim'):
                    # 循环一致性生成器
                    fake_real = model.sim2real(sim_points, real_points)
                    fake_sim = model.real2sim(real_points, sim_points)
                    
                    # 循环重建
                    cycled_sim = model.real2sim(fake_real, sim_points)
                    cycled_real = model.sim2real(fake_sim, real_points)
                    
                    # 身份映射
                    identity_real = model.sim2real(real_points, real_points)
                    identity_sim = model.real2sim(sim_points, sim_points)
                    
                else:
                    # 简单生成器
                    fake_real = model(sim_points, real_points)
                    fake_sim = None
                    cycled_sim = None
                    cycled_real = None
                    identity_real = None
                    identity_sim = None
                
                processing_time = time.time() - start_time
                all_results['processing_times'].append(processing_time)
                
                # 评估生成质量
                gen_quality = self.evaluate_generation_quality(fake_real, real_points)
                all_results['generation_quality'].append(gen_quality)
                
                # 评估内容保持
                content_preservation = self.evaluate_content_preservation(fake_real, sim_points)
                all_results['content_preservation'].append(content_preservation)
                
                # 评估风格迁移效果
                style_transfer = self.evaluate_style_transfer_effectiveness(
                    sim_points, fake_real, real_points
                )
                all_results['style_transfer'].append(style_transfer)
                
                # 评估循环一致性（如果可用）
                if cycled_sim is not None:
                    cycle_consistency = self.evaluate_cycle_consistency(sim_points, cycled_sim)
                    all_results['cycle_consistency'].append(cycle_consistency)
                
                # 评估身份保持（如果可用）
                if identity_real is not None:
                    identity_preservation = self.evaluate_identity_preservation(real_points, identity_real)
                    all_results['identity_preservation'].append(identity_preservation)
                
                # 保存可视化例子
                if save_examples and example_count < num_examples:
                    self._save_evaluation_example(
                        sim_points[0], fake_real[0], real_points[0],
                        example_count, batch_idx
                    )
                    example_count += 1
        
        # 计算汇总统计
        summary_results = self._compute_summary_statistics(all_results)
        
        # 保存结果
        self._save_evaluation_results(summary_results, all_results)
        
        return summary_results
    
    def _compute_summary_statistics(self, all_results: Dict[str, List]) -> Dict[str, Any]:
        """计算汇总统计"""
        summary = {}
        
        for category, results_list in all_results.items():
            if not results_list:
                continue
            
            if category == 'processing_times':
                summary[category] = {
                    'mean': np.mean(results_list),
                    'std': np.std(results_list),
                    'min': np.min(results_list),
                    'max': np.max(results_list),
                    'total_samples': len(results_list)
                }
            else:
                # 计算每个指标的统计
                category_summary = {}
                
                # 收集所有指标名称
                all_metrics = set()
                for result in results_list:
                    all_metrics.update(result.keys())
                
                # 计算每个指标的统计
                for metric in all_metrics:
                    values = [result.get(metric, 0) for result in results_list if metric in result]
                    if values:
                        category_summary[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'median': np.median(values)
                        }
                
                summary[category] = category_summary
        
        return summary
    
    def _save_evaluation_example(self, sim_points: torch.Tensor,
                                generated: torch.Tensor,
                                real_points: torch.Tensor,
                                example_idx: int,
                                batch_idx: int):
        """保存评估例子的可视化"""
        examples_dir = os.path.join(self.output_dir, 'examples')
        os.makedirs(examples_dir, exist_ok=True)
        
        # 转换为numpy
        sim_np = sim_points.cpu().numpy()
        gen_np = generated.cpu().numpy()
        real_np = real_points.cpu().numpy()
        
        # 保存对比图
        comparison_path = os.path.join(examples_dir, f'example_{example_idx}_comparison.png')
        self.visualizer.plot_style_transfer_result(
            sim_np, gen_np, real_np,
            title=f'Style Transfer Example {example_idx + 1}',
            save_path=comparison_path
        )
        
        # 保存点云文件
        np.save(os.path.join(examples_dir, f'example_{example_idx}_sim.npy'), sim_np)
        np.save(os.path.join(examples_dir, f'example_{example_idx}_generated.npy'), gen_np)
        np.save(os.path.join(examples_dir, f'example_{example_idx}_real.npy'), real_np)
    
    def _save_evaluation_results(self, summary_results: Dict, detailed_results: Dict):
        """保存评估结果"""
        # 保存汇总结果
        summary_path = os.path.join(self.output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        # 保存详细结果
        detailed_path = os.path.join(self.output_dir, 'evaluation_detailed.json')
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # 生成人类可读的报告
        self._generate_human_readable_report(summary_results)
    
    def _generate_human_readable_report(self, summary_results: Dict):
        """生成人类可读的评估报告"""
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("POINT CLOUD STYLE TRANSFER EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # 处理时间
            if 'processing_times' in summary_results:
                times = summary_results['processing_times']
                f.write("PROCESSING PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average processing time: {times['mean']:.4f}s\n")
                f.write(f"Throughput: {1/times['mean']:.2f} samples/second\n")
                f.write(f"Total samples processed: {times['total_samples']}\n\n")
            
            # 生成质量
            if 'generation_quality' in summary_results:
                gen_quality = summary_results['generation_quality']
                f.write("GENERATION QUALITY\n")
                f.write("-" * 30 + "\n")
                
                if 'chamfer_distance' in gen_quality:
                    cd = gen_quality['chamfer_distance']
                    f.write(f"Chamfer Distance: {cd['mean']:.6f} ± {cd['std']:.6f}\n")
                
                if 'coverage_score' in gen_quality:
                    coverage = gen_quality['coverage_score']
                    f.write(f"Coverage Score: {coverage['mean']:.4f} ± {coverage['std']:.4f}\n")
                
                if 'uniformity_score' in gen_quality:
                    uniformity = gen_quality['uniformity_score']
                    f.write(f"Uniformity Score: {uniformity['mean']:.4f} ± {uniformity['std']:.4f}\n")
                
                f.write("\n")
            
            # 风格迁移效果
            if 'style_transfer' in summary_results:
                style = summary_results['style_transfer']
                f.write("STYLE TRANSFER EFFECTIVENESS\n")
                f.write("-" * 30 + "\n")
                
                if 'style_transfer_ratio' in style:
                    ratio = style['style_transfer_ratio']
                    f.write(f"Style Transfer Ratio: {ratio['mean']:.4f} ± {ratio['std']:.4f}\n")
                
                if 'style_feature_similarity' in style:
                    similarity = style['style_feature_similarity']
                    f.write(f"Style Feature Similarity: {similarity['mean']:.4f} ± {similarity['std']:.4f}\n")
                
                f.write("\n")
            
            # 循环一致性
            if 'cycle_consistency' in summary_results:
                cycle = summary_results['cycle_consistency']
                f.write("CYCLE CONSISTENCY\n")
                f.write("-" * 30 + "\n")
                
                if 'cycle_consistency_error' in cycle:
                    error = cycle['cycle_consistency_error']
                    f.write(f"Cycle Consistency Error: {error['mean']:.6f} ± {error['std']:.6f}\n")
                
                f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            
            # 基于结果生成建议
            recommendations = self._generate_recommendations(summary_results)
            for rec in recommendations:
                f.write(f"• {rec}\n")
        
        print(f"Evaluation report saved to: {report_path}")
    
    def _generate_recommendations(self, summary_results: Dict) -> List[str]:
        """基于评估结果生成改进建议"""
        recommendations = []
        
        # 检查生成质量
        if 'generation_quality' in summary_results:
            gen_quality = summary_results['generation_quality']
            
            if 'chamfer_distance' in gen_quality:
                cd_mean = gen_quality['chamfer_distance']['mean']
                if cd_mean > 0.01:
                    recommendations.append("Consider improving generation quality - Chamfer distance is relatively high")
            
            if 'coverage_score' in gen_quality:
                coverage_mean = gen_quality['coverage_score']['mean']
                if coverage_mean < 0.8:
                    recommendations.append("Improve point cloud coverage - some areas may not be well represented")
        
        # 检查风格迁移效果
        if 'style_transfer' in summary_results:
            style = summary_results['style_transfer']
            
            if 'style_transfer_ratio' in style:
                ratio_mean = style['style_transfer_ratio']['mean']
                if ratio_mean < 0.5:
                    recommendations.append("Style transfer effectiveness could be improved - consider adjusting loss weights")
        
        # 检查循环一致性
        if 'cycle_consistency' in summary_results:
            cycle = summary_results['cycle_consistency']
            
            if 'cycle_consistency_error' in cycle:
                error_mean = cycle['cycle_consistency_error']['mean']
                if error_mean > 0.02:
                    recommendations.append("Cycle consistency error is high - consider increasing cycle loss weight")
        
        # 检查处理效率
        if 'processing_times' in summary_results:
            times = summary_results['processing_times']
            throughput = 1 / times['mean']
            if throughput < 1.0:
                recommendations.append("Processing speed could be optimized for better throughput")
        
        if not recommendations:
            recommendations.append("Model performance looks good across all metrics!")
        
        return recommendations


def load_model_for_evaluation(model_path: str, config: Any, device: torch.device):
    """
    加载模型用于评估
    Args:
        model_path: 模型检查点路径
        config: 配置对象
        device: 设备
    Returns:
        加载的模型
    """
    from models.generator import CycleConsistentGenerator
    
    # 创建模型
    model = CycleConsistentGenerator(
        input_channels=config.input_dim,
        feature_channels=config.pointnet_channels,
        style_dim=config.generator_dim,
        latent_dim=config.latent_dim,
        num_points=config.chunk_size
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    
    return model