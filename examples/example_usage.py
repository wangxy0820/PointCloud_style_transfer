"""
完整的使用示例
"""

import numpy as np
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.preprocessing import ImprovedPointCloudPreprocessor
from scripts.inference import DiffusionInference
from utils.visualization import PointCloudVisualizer


def example_preprocessing():
    """数据预处理示例"""
    print("=== Data Preprocessing Example ===")
    
    # 创建预处理器
    preprocessor = ImprovedPointCloudPreprocessor(
        total_points=120000,
        chunk_size=2048,
        overlap_ratio=0.3
    )
    
    # 模拟数据
    sim_points = np.random.randn(120000, 3).astype(np.float32)
    real_points = np.random.randn(120000, 3).astype(np.float32)
    
    # 预处理并保存
    output_dir = "temp_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = preprocessor.save_preprocessed_data(
        sim_points, real_points, output_dir, "example_pair"
    )
    
    print(f"Preprocessed data saved to: {save_path}")
    
    # 加载并检查
    data = torch.load(save_path)
    print(f"Number of simulation chunks: {len(data['sim_chunks'])}")
    print(f"Number of real chunks: {len(data['real_chunks'])}")
    print(f"Chunk size: {data['sim_chunks'][0][0].shape}")


def example_inference():
    """推理示例"""
    print("\n=== Inference Example ===")
    
    # 假设已有训练好的模型
    checkpoint_path = "checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print("Note: This is a demo. Please train a model first.")
        print("Creating dummy checkpoint for demonstration...")
        
        # 创建假的检查点用于演示
        config = Config()
        checkpoint = {
            'config': config,
            'model_state_dict': {},
            'style_encoder_state_dict': {},
            'epoch': 100
        }
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
    
    # 创建推理器
    inference = DiffusionInference(checkpoint_path, device='cuda')
    
    # 加载测试数据
    sim_points = np.random.randn(120000, 3).astype(np.float32)
    real_reference = np.random.randn(50000, 3).astype(np.float32)
    
    print("Transferring style...")
    # 执行风格转换
    # transferred = inference.transfer_style(sim_points, real_reference)
    # print(f"Transferred point cloud shape: {transferred.shape}")
    
    print("Note: Full inference requires a trained model.")


def example_visualization():
    """可视化示例"""
    print("\n=== Visualization Example ===")
    
    # 创建可视化器
    visualizer = PointCloudVisualizer()
    
    # 创建示例数据
    n_points = 5000
    
    # 仿真点云（球形）
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.normal(1.0, 0.1, n_points)
    
    sim_points = np.stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ], axis=1)
    
    # 生成的点云（添加一些噪声和变形）
    generated_points = sim_points + np.random.normal(0, 0.05, sim_points.shape)
    generated_points[:, 2] *= 0.8  # 压缩Z轴
    
    # 真实参考（立方体）
    real_points = np.random.uniform(-1, 1, (n_points, 3))
    
    # 创建可视化
    print("Creating visualization...")
    visualizer.plot_style_transfer_result(
        sim_points,
        generated_points,
        real_points,
        title="Style Transfer Example",
        save_path="example_visualization.png"
    )
    print("Visualization saved to: example_visualization.png")


def example_batch_processing():
    """批处理示例"""
    print("\n=== Batch Processing Example ===")
    
    # 处理多个文件的示例代码
    input_folder = "datasets/simulation"
    output_folder = "datasets/transferred"
    
    print("Example batch processing code:")
    print("""
    # 创建推理器
    inference = DiffusionInference('checkpoints/best_model.pth')
    
    # 加载真实参考
    real_reference = np.load('datasets/real_world/reference.npy')
    
    # 处理文件夹
    inference.process_folder(
        sim_folder='datasets/simulation',
        real_reference_path='datasets/real_world/reference.npy',
        output_folder='datasets/transferred'
    )
    """)


def example_custom_training():
    """自定义训练示例"""
    print("\n=== Custom Training Example ===")
    
    print("Example custom training configuration:")
    print("""
    from config.config import Config
    from training.progressive_trainer import ProgressiveDiffusionTrainer
    
    # 自定义配置
    config = Config()
    config.batch_size = 16
    config.num_epochs = 200
    config.learning_rate = 0.0002
    config.progressive_training = True
    config.initial_chunks = 5
    
    # 创建渐进式训练器
    trainer = ProgressiveDiffusionTrainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader)
    """)


def main():
    """运行所有示例"""
    print("Point Cloud Style Transfer - Usage Examples")
    print("="*50)
    
    # 运行各个示例
    example_preprocessing()
    example_inference()
    example_visualization()
    example_batch_processing()
    example_custom_training()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("For full functionality, please train a model using:")
    print("  python scripts/train.py --data_dir datasets/processed")


if __name__ == "__main__":
    main()

