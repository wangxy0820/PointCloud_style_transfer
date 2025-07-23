"""
结果可视化脚本
"""

import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import PointCloudVisualizer


def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud results')
    parser.add_argument('--original', type=str, required=True, help='Original point cloud')
    parser.add_argument('--generated', type=str, required=True, help='Generated point cloud')
    parser.add_argument('--reference', type=str, help='Reference point cloud')
    parser.add_argument('--output_path', type=str, default='visualization.png')
    parser.add_argument('--sample_size', type=int, default=5000, help='Points to visualize')
    parser.add_argument('--interactive', action='store_true', help='Interactive 3D view')
    
    args = parser.parse_args()
    
    # 加载点云
    original = np.load(args.original)
    generated = np.load(args.generated)
    
    visualizer = PointCloudVisualizer()
    
    if args.reference:
        reference = np.load(args.reference)
        
        if args.interactive:
            # 交互式可视化
            visualizer.visualize_interactive(
                [original, generated, reference],
                ['Original', 'Generated', 'Reference'],
                [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]
            )
        else:
            # 静态图像
            visualizer.plot_style_transfer_result(
                original, generated, reference,
                title='Point Cloud Style Transfer Result',
                save_path=args.output_path,
                sample_size=args.sample_size
            )
    else:
        # 只有两个点云
        if args.interactive:
            visualizer.visualize_interactive(
                [original, generated],
                ['Original', 'Generated'],
                [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]
            )
        else:
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=(10, 5))
            
            # 采样
            if len(original) > args.sample_size:
                idx = np.random.choice(len(original), args.sample_size, replace=False)
                orig_sample = original[idx]
                gen_sample = generated[idx]
            else:
                orig_sample = original
                gen_sample = generated
            
            # 原始点云
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(orig_sample[:, 0], orig_sample[:, 1], orig_sample[:, 2],
                       c=[0.1, 0.1, 0.8], s=1, alpha=0.5)
            ax1.set_title('Original')
            
            # 生成的点云
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(gen_sample[:, 0], gen_sample[:, 1], gen_sample[:, 2],
                       c=[0.1, 0.8, 0.1], s=1, alpha=0.5)
            ax2.set_title('Generated')
            
            plt.tight_layout()
            plt.savefig(args.output_path, dpi=150)
            print(f"Visualization saved to: {args.output_path}")


if __name__ == "__main__":
    main()
