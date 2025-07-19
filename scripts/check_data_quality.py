#!/usr/bin/env python3
"""
数据质量检查工具
检查点云数据的质量问题并生成报告
"""

import argparse
import os
import sys
import numpy as np
import glob
import json
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import load_point_cloud, compute_point_cloud_statistics, remove_outliers


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Point Cloud Data Quality Checker')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing point cloud data')
    parser.add_argument('--output_dir', type=str, default='data_quality_report',
                       help='Directory to save quality report')
    parser.add_argument('--file_pattern', type=str, default='*.npy',
                       help='File pattern to match')
    parser.add_argument('--max_files', type=int, default=0,
                       help='Maximum number of files to check (0 = all)')
    
    # 检查选项
    parser.add_argument('--check_duplicates', action='store_true',
                       help='Check for duplicate points')
    parser.add_argument('--check_outliers', action='store_true',
                       help='Check for outlier points')
    parser.add_argument('--check_empty', action='store_true',
                       help='Check for empty or nearly empty files')
    parser.add_argument('--check_format', action='store_true',
                       help='Check file format consistency')
    parser.add_argument('--check_statistics', action='store_true',
                       help='Compute detailed statistics')
    
    # 阈值参数
    parser.add_argument('--min_points', type=int, default=1000,
                       help='Minimum number of points for valid file')
    parser.add_argument('--max_points', type=int, default=1000000,
                       help='Maximum number of points for valid file')
    parser.add_argument('--duplicate_tolerance', type=float, default=1e-6,
                       help='Tolerance for duplicate point detection')
    parser.add_argument('--outlier_std_threshold', type=float, default=3.0,
                       help='Standard deviation threshold for outlier detection')
    
    # 输出选项
    parser.add_argument('--save_plots', action='store_true',
                       help='Save statistical plots')
    parser.add_argument('--save_detailed_report', action='store_true',
                       help='Save detailed per-file report')
    parser.add_argument('--fix_issues', action='store_true',
                       help='Attempt to fix detected issues')
    
    return parser.parse_args()


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, args):
        self.args = args
        self.issues = defaultdict(list)
        self.statistics = defaultdict(list)
        self.file_reports = {}
    
    def check_file_format(self, file_path: str) -> dict:
        """检查文件格式"""
        report = {'file_path': file_path, 'issues': [], 'valid': True}
        
        try:
            points = load_point_cloud(file_path)
            
            # 检查数据类型
            if points.dtype != np.float32:
                report['issues'].append(f"Data type is {points.dtype}, expected float32")
                report['valid'] = False
            
            # 检查维度
            if len(points.shape) != 2:
                report['issues'].append(f"Shape is {points.shape}, expected 2D array")
                report['valid'] = False
            elif points.shape[1] != 3:
                if points.shape[1] < 3:
                    report['issues'].append(f"Only {points.shape[1]} coordinates, expected at least 3")
                    report['valid'] = False
                else:
                    report['issues'].append(f"Extra coordinates: {points.shape[1]}, will use first 3")
            
            # 检查是否包含NaN或Inf
            if np.any(np.isnan(points)):
                nan_count = np.sum(np.isnan(points))
                report['issues'].append(f"Contains {nan_count} NaN values")
                report['valid'] = False
            
            if np.any(np.isinf(points)):
                inf_count = np.sum(np.isinf(points))
                report['issues'].append(f"Contains {inf_count} infinite values")
                report['valid'] = False
            
            report['num_points'] = len(points)
            report['shape'] = points.shape
            report['dtype'] = str(points.dtype)
            
        except Exception as e:
            report['issues'].append(f"Failed to load file: {e}")
            report['valid'] = False
        
        return report
    
    def check_point_count(self, points: np.ndarray, file_path: str) -> dict:
        """检查点数"""
        report = {'issues': [], 'valid': True}
        num_points = len(points)
        
        if num_points < self.args.min_points:
            report['issues'].append(f"Too few points: {num_points} < {self.args.min_points}")
            report['valid'] = False
        
        if num_points > self.args.max_points:
            report['issues'].append(f"Too many points: {num_points} > {self.args.max_points}")
            report['valid'] = False
        
        return report
    
    def check_duplicates(self, points: np.ndarray) -> dict:
        """检查重复点"""
        report = {'issues': [], 'valid': True}
        
        # 使用容差检查重复点
        tolerance = self.args.duplicate_tolerance
        rounded_points = np.round(points / tolerance).astype(int)
        unique_points, unique_indices = np.unique(rounded_points, axis=0, return_index=True)
        
        num_duplicates = len(points) - len(unique_points)
        duplicate_ratio = num_duplicates / len(points)
        
        if duplicate_ratio > 0.01:  # 超过1%重复
            report['issues'].append(f"High duplicate ratio: {duplicate_ratio:.2%} ({num_duplicates} duplicates)")
            report['valid'] = False
        
        report['num_duplicates'] = num_duplicates
        report['duplicate_ratio'] = duplicate_ratio
        
        return report
    
    def check_outliers(self, points: np.ndarray) -> dict:
        """检查离群点"""
        report = {'issues': [], 'valid': True}
        
        if len(points) < 10:
            return report
        
        # 使用统计方法检测离群点
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + self.args.outlier_std_threshold * std_dist
        
        outlier_mask = distances > threshold
        num_outliers = np.sum(outlier_mask)
        outlier_ratio = num_outliers / len(points)
        
        if outlier_ratio > 0.05:  # 超过5%离群点
            report['issues'].append(f"High outlier ratio: {outlier_ratio:.2%} ({num_outliers} outliers)")
            report['valid'] = False
        
        report['num_outliers'] = num_outliers
        report['outlier_ratio'] = outlier_ratio
        report['mean_distance_to_centroid'] = mean_dist
        report['std_distance_to_centroid'] = std_dist
        
        return report
    
    def compute_statistics(self, points: np.ndarray) -> dict:
        """计算详细统计"""
        stats = compute_point_cloud_statistics(points)
        
        # 添加一些额外的统计量
        ranges = stats['max'] - stats['min']
        stats['ranges'] = ranges
        stats['volume_estimate'] = np.prod(ranges)
        
        # 计算点云的"质量"指标
        if 'density_estimate' in stats:
            stats['quality_score'] = min(1.0, stats['density_estimate'] / 1000.0)
        else:
            stats['quality_score'] = 0.5
        
        return stats
    
    def check_single_file(self, file_path: str) -> dict:
        """检查单个文件"""
        file_report = {
            'file_path': file_path,
            'overall_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # 格式检查
        format_report = self.check_file_format(file_path)
        file_report.update(format_report)
        
        if not format_report['valid']:
            file_report['overall_valid'] = False
            return file_report
        
        # 加载点云
        try:
            points = load_point_cloud(file_path)[:, :3]  # 只保留xyz
        except Exception as e:
            file_report['issues'].append(f"Failed to load: {e}")
            file_report['overall_valid'] = False
            return file_report
        
        # 点数检查
        count_report = self.check_point_count(points, file_path)
        if not count_report['valid']:
            file_report['overall_valid'] = False
        file_report['issues'].extend(count_report['issues'])
        
        # 重复点检查
        if self.args.check_duplicates:
            dup_report = self.check_duplicates(points)
            if not dup_report['valid']:
                file_report['warnings'].extend(dup_report['issues'])
            file_report['duplicate_info'] = {
                'num_duplicates': dup_report['num_duplicates'],
                'duplicate_ratio': dup_report['duplicate_ratio']
            }
        
        # 离群点检查
        if self.args.check_outliers:
            outlier_report = self.check_outliers(points)
            if not outlier_report['valid']:
                file_report['warnings'].extend(outlier_report['issues'])
            file_report['outlier_info'] = {
                'num_outliers': outlier_report['num_outliers'],
                'outlier_ratio': outlier_report['outlier_ratio']
            }
        
        # 统计信息
        if self.args.check_statistics:
            stats = self.compute_statistics(points)
            file_report['statistics'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in stats.items()
            }
        
        return file_report
    
    def fix_file_issues(self, file_path: str, issues: list) -> bool:
        """尝试修复文件问题"""
        if not self.args.fix_issues:
            return False
        
        try:
            points = load_point_cloud(file_path)
            fixed = False
            
            # 修复数据类型
            if points.dtype != np.float32:
                points = points.astype(np.float32)
                fixed = True
            
            # 修复维度
            if points.shape[1] > 3:
                points = points[:, :3]
                fixed = True
            
            # 移除NaN和Inf
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                valid_mask = np.isfinite(points).all(axis=1)
                points = points[valid_mask]
                fixed = True
            
            # 移除重复点
            if self.args.check_duplicates:
                original_len = len(points)
                tolerance = self.args.duplicate_tolerance
                rounded_points = np.round(points / tolerance)
                _, unique_indices = np.unique(rounded_points, axis=0, return_index=True)
                points = points[unique_indices]
                if len(points) < original_len:
                    fixed = True
            
            # 移除离群点
            if self.args.check_outliers:
                original_len = len(points)
                points = remove_outliers(points, std_threshold=self.args.outlier_std_threshold)
                if len(points) < original_len:
                    fixed = True
            
            # 保存修复后的文件
            if fixed:
                backup_path = file_path + '.backup'
                os.rename(file_path, backup_path)
                np.save(file_path, points)
                print(f"Fixed issues in {file_path} (backup saved as {backup_path})")
                return True
            
        except Exception as e:
            print(f"Failed to fix {file_path}: {e}")
        
        return False
    
    def generate_plots(self, output_dir: str):
        """生成统计图表"""
        if not self.args.save_plots:
            return
        
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 收集所有文件的统计数据
        all_stats = []
        for report in self.file_reports.values():
            if 'statistics' in report and report['overall_valid']:
                all_stats.append(report['statistics'])
        
        if not all_stats:
            return
        
        # 点数分布
        point_counts = [report['num_points'] for report in self.file_reports.values() 
                       if 'num_points' in report]
        if point_counts:
            plt.figure(figsize=(10, 6))
            plt.hist(point_counts, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Number of Points')
            plt.ylabel('Frequency')
            plt.title('Distribution of Point Counts')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'point_count_distribution.png'), dpi=300)
            plt.close()
        
        # 质量分数分布
        quality_scores = [stats.get('quality_score', 0) for stats in all_stats]
        if quality_scores:
            plt.figure(figsize=(10, 6))
            plt.hist(quality_scores, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Quality Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Quality Scores')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'quality_score_distribution.png'), dpi=300)
            plt.close()
        
        # 重复点比例分布
        if self.args.check_duplicates:
            dup_ratios = [report.get('duplicate_info', {}).get('duplicate_ratio', 0) 
                         for report in self.file_reports.values()]
            dup_ratios = [r for r in dup_ratios if r > 0]
            
            if dup_ratios:
                plt.figure(figsize=(10, 6))
                plt.hist(dup_ratios, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Duplicate Ratio')
                plt.ylabel('Frequency')
                plt.title('Distribution of Duplicate Ratios')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, 'duplicate_ratio_distribution.png'), dpi=300)
                plt.close()
    
    def generate_report(self, output_dir: str):
        """生成质量报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 汇总统计
        total_files = len(self.file_reports)
        valid_files = sum(1 for r in self.file_reports.values() if r['overall_valid'])
        invalid_files = total_files - valid_files
        
        total_issues = sum(len(r['issues']) for r in self.file_reports.values())
        total_warnings = sum(len(r['warnings']) for r in self.file_reports.values())
        
        # 创建汇总报告
        summary = {
            'total_files': total_files,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'validity_rate': valid_files / total_files if total_files > 0 else 0,
            'total_issues': total_issues,
            'total_warnings': total_warnings,
            'common_issues': self._get_common_issues(),
            'recommendations': self._get_recommendations()
        }
        
        # 保存汇总报告
        with open(os.path.join(output_dir, 'quality_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存详细报告
        if self.args.save_detailed_report:
            with open(os.path.join(output_dir, 'detailed_report.json'), 'w') as f:
                json.dump(self.file_reports, f, indent=2)
        
        # 生成图表
        self.generate_plots(output_dir)
        
        # 打印汇总
        print("\n" + "="*60)
        print("DATA QUALITY REPORT SUMMARY")
        print("="*60)
        print(f"Total files checked: {total_files}")
        print(f"Valid files: {valid_files} ({summary['validity_rate']:.1%})")
        print(f"Invalid files: {invalid_files}")
        print(f"Total issues found: {total_issues}")
        print(f"Total warnings: {total_warnings}")
        print(f"Report saved to: {output_dir}")
        print("="*60)
    
    def _get_common_issues(self) -> dict:
        """获取常见问题统计"""
        issue_counts = defaultdict(int)
        
        for report in self.file_reports.values():
            for issue in report['issues']:
                # 简化问题描述以便统计
                if 'NaN' in issue:
                    issue_counts['NaN values'] += 1
                elif 'infinite' in issue:
                    issue_counts['Infinite values'] += 1
                elif 'duplicate' in issue.lower():
                    issue_counts['High duplicate ratio'] += 1
                elif 'outlier' in issue.lower():
                    issue_counts['High outlier ratio'] += 1
                elif 'few points' in issue:
                    issue_counts['Too few points'] += 1
                elif 'many points' in issue:
                    issue_counts['Too many points'] += 1
                else:
                    issue_counts['Other'] += 1
        
        return dict(issue_counts)
    
    def _get_recommendations(self) -> list:
        """获取改进建议"""
        recommendations = []
        
        common_issues = self._get_common_issues()
        
        if common_issues.get('NaN values', 0) > 0:
            recommendations.append("Remove or fix NaN values in point cloud data")
        
        if common_issues.get('High duplicate ratio', 0) > 0:
            recommendations.append("Remove duplicate points to improve data quality")
        
        if common_issues.get('High outlier ratio', 0) > 0:
            recommendations.append("Consider outlier removal or data cleaning")
        
        if common_issues.get('Too few points', 0) > 0:
            recommendations.append("Increase point density or filter out low-quality files")
        
        if common_issues.get('Too many points', 0) > 0:
            recommendations.append("Consider down-sampling very large point clouds")
        
        return recommendations


def main():
    """主函数"""
    args = parse_args()
    
    # 获取文件列表
    if os.path.isdir(args.data_dir):
        files = glob.glob(os.path.join(args.data_dir, "**", args.file_pattern), recursive=True)
    else:
        print(f"Error: {args.data_dir} is not a directory")
        return
    
    if not files:
        print(f"No files found matching pattern {args.file_pattern} in {args.data_dir}")
        return
    
    # 限制文件数量
    if args.max_files > 0:
        files = files[:args.max_files]
    
    print(f"Checking quality of {len(files)} files...")
    
    # 创建检查器
    checker = DataQualityChecker(args)
    
    # 检查每个文件
    for file_path in tqdm(files, desc="Checking files"):
        report = checker.check_single_file(file_path)
        checker.file_reports[file_path] = report
        
        # 尝试修复问题
        if not report['overall_valid'] and args.fix_issues:
            checker.fix_file_issues(file_path, report['issues'])
    
    # 生成报告
    checker.generate_report(args.output_dir)


if __name__ == "__main__":
    main()