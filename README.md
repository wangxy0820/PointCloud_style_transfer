# 点云风格迁移项目使用指南

## 项目概述

这是一个基于PointNet++和GAN的点云风格迁移系统，能够将simulation域的点云数据转换为具有real world风格的点云数据。项目支持12万个点的大规模点云处理，采用分块策略和循环一致性训练。

## 技术特点

- **PointNet++特征提取**: 改进的PointNet++网络提取层次化特征
- **分块处理策略**: 支持12万点大规模点云分块处理（5K-10K点每块）
- **循环一致性GAN**: Sim2Real和Real2Sim双向风格迁移
- **多尺度判别**: 混合判别器结合全局和局部判别
- **渐进式训练**: 支持warmup和学习率调度
- **完整评估体系**: Chamfer距离、EMD、FPD等多种评估指标

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone [your-repo-url]
cd pointcloud_style_transfer

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 2. 数据准备

将你的点云数据按以下结构组织：

```
datasets/
├── simulation/          # 仿真点云数据(.npy文件)
│   ├── sim_001.npy
│   ├── sim_002.npy
│   └── ...
└── real_world/          # 真实点云数据(.npy文件)
    ├── real_001.npy
    ├── real_002.npy
    └── ...
```

**注意**: 每个.npy文件应包含形状为[N, 3]的点云数据，其中N约为12万个点。

### 3. 数据预处理

```bash
# 预处理数据，将12万点分块为8192点的小块
python data/preprocess.py \
    --sim_dir datasets/simulation \
    --real_dir datasets/real_world \
    --output_dir datasets/processed \
    --chunk_size 8192 \
    --chunk_method spatial
```

预处理选项：
- `--chunk_method`: 分块方法
  - `spatial`: 基于空间聚类分块（推荐）
  - `random`: 随机采样分块
  - `sliding`: 滑动窗口分块

### 4. 模型训练

```bash
# 基础训练命令
python scripts/train.py \
    --data_dir datasets/processed \
    --batch_size 8 \
    --num_epochs 200 \
    --learning_rate_g 0.0002 \
    --learning_rate_d 0.0001 \
    --experiment_name my_style_transfer

# 使用数据增强
python scripts/train.py \
    --data_dir datasets/processed \
    --batch_size 8 \
    --num_epochs 200 \
    --use_augmentation \
    --rotation_range 0.1 \
    --jitter_std 0.01 \
    --experiment_name augmented_training

# 从检查点恢复训练
python scripts/train.py \
    --data_dir datasets/processed \
    --batch_size 8 \
    --resume experiments/my_style_transfer/checkpoints/latest.pth
```

关键训练参数：
- `--lambda_recon 10.0`: 重建损失权重
- `--lambda_adv 1.0`: 对抗损失权重  
- `--lambda_cycle 5.0`: 循环一致性损失权重
- `--lambda_identity 2.0`: 身份损失权重

### 5. 模型测试

```bash
# 在测试集上评估模型
python scripts/test.py \
    --model_path experiments/my_style_transfer/checkpoints/best_model.pth \
    --data_dir datasets/processed \
    --output_dir test_results \
    --compute_all_metrics \
    --save_visualizations \
    --direction sim2real

# 保存生成的点云
python scripts/test.py \
    --model_path experiments/my_style_transfer/checkpoints/best_model.pth \
    --data_dir datasets/processed \
    --output_dir test_results \
    --save_generated \
    --save_metrics_csv
```

### 6. 推理生成

```bash
# 对新的仿真点云进行风格迁移
python scripts/inference.py \
    --model_path experiments/my_style_transfer/checkpoints/best_model.pth \
    --input_dir new_simulation_data \
    --output_dir generated_real_style \
    --style_reference datasets/real_world/real_001.npy \
    --direction sim2real \
    --preprocess \
    --merge_chunks

# 使用多个风格参考
python scripts/inference.py \
    --model_path experiments/my_style_transfer/checkpoints/best_model.pth \
    --input_dir new_simulation_data \
    --output_dir generated_results \
    --style_dir datasets/real_world \
    --random_style \
    --create_visualization
```

### 7. 结果可视化

```bash
# 可视化训练结果
python scripts/visualize.py \
    --input_dir experiments/my_style_transfer/results \
    --output_dir visualizations \
    --mode style_transfer \
    --save_html

# 对比原始和生成的点云
python scripts/visualize.py \
    --mode comparison \
    --original_dir new_simulation_data \
    --generated_dir generated_real_style \
    --output_dir comparison_vis \
    --max_files 10

# 可视化训练曲线
python scripts/visualize.py \
    --mode training_curves \
    --log_dir experiments/my_style_transfer/logs \
    --output_dir training_analysis

# 可视化评估指标
python scripts/visualize.py \
    --mode metrics \
    --input_dir test_results \
    --output_dir metrics_analysis
```

## 项目文件结构

```
pointcloud_style_transfer/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── config/
│   └── config.py            # 配置文件
├── data/
│   ├── dataset.py           # 数据集类
│   ├── preprocess.py        # 数据预处理
│   └── utils.py             # 数据工具
├── models/
│   ├── pointnet2.py         # PointNet++模型
│   ├── generator.py         # 生成器
│   ├── discriminator.py     # 判别器
│   └── losses.py            # 损失函数
├── training/
│   ├── trainer.py           # 训练器
│   └── utils.py             # 训练工具
├── evaluation/
│   ├── metrics.py           # 评估指标
│   └── evaluator.py         # 评估器
├── visualization/
│   ├── visualize.py         # 可视化工具
│   └── plot_utils.py        # 绘图工具
├── scripts/
│   ├── train.py             # 训练脚本
│   ├── test.py              # 测试脚本
│   ├── inference.py         # 推理脚本
│   └── visualize.py         # 可视化脚本
├── experiments/             # 实验结果目录
├── datasets/                # 数据集目录
└── logs/                    # 日志目录
```

## 评估指标

### 几何质量指标
- **Chamfer Distance (CD)**: 两个点云之间的几何相似性
- **Earth Mover's Distance (EMD)**: 点云分布差异
- **Hausdorff Distance**: 最大最小距离
- **Minimum Matching Distance**: 最优匹配距离

### 风格迁移指标
- **Fréchet Point Cloud Distance (FPD)**: 基于特征的质量评估
- **Coverage Score**: 覆盖度评分
- **Uniformity Score**: 点云均匀性
- **Style Transfer Ratio**: 风格迁移效果比率

## 常见问题解决

### 1. GPU内存不足
```bash
# 减小批次大小
python scripts/train.py --batch_size 4

# 减小分块大小
python data/preprocess.py --chunk_size 4096
```

### 2. 训练不收敛
```bash
# 调整学习率
python scripts/train.py \
    --learning_rate_g 0.0001 \
    --learning_rate_d 0.00005

# 调整损失权重
python scripts/train.py \
    --lambda_cycle 10.0 \
    --lambda_identity 5.0
```

### 3. 生成质量不佳
```bash
# 增加预热轮数
python scripts/train.py --warmup_epochs 20

# 使用更大的分块
python data/preprocess.py --chunk_size 10240
```

### 4. 数据加载慢
```bash
# 增加工作进程
python scripts/train.py --num_workers 8

# 使用内存固定
python scripts/train.py --pin_memory
```

## 高级使用

### 自定义损失函数
在`models/losses.py`中修改`StyleTransferLoss`类，调整损失权重：

```python
# 修改损失权重
self.lambda_recon = 15.0    # 增强重建质量
self.lambda_cycle = 8.0     # 增强循环一致性
```

### 自定义网络架构
在`models/pointnet2.py`中修改网络结构：

```python
# 调整特征通道
feature_channels = [128, 256, 512, 1024]  # 更大的网络

# 调整潜在维度
latent_dim = 1024  # 更高维的特征表示
```

### 分布式训练
```bash
# 多GPU训练（需要修改训练脚本）
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py \
    --batch_size 32 \
    --num_workers 16
```

## 结果分析

### 训练监控
- TensorBoard日志：`tensorboard --logdir experiments/[experiment_name]/logs`
- 损失曲线：查看生成器和判别器损失变化
- 验证指标：关注Chamfer距离和EMD变化

### 质量评估
- **CD < 0.01**: 优秀的几何保持
- **Style Transfer Ratio > 0.7**: 良好的风格迁移
- **Coverage Score > 0.8**: 充分的点云覆盖

### 输出文件
- `best_model.pth`: 最佳模型权重
- `metrics.json`: 详细评估指标
- `generated_*.npy`: 生成的点云文件
- `visualizations/`: 可视化结果图片

## 扩展开发

项目采用模块化设计，支持以下扩展：

1. **新的网络架构**: 在`models/`目录添加新模型
2. **新的损失函数**: 在`models/losses.py`中添加
3. **新的评估指标**: 在`evaluation/metrics.py`中添加
4. **新的数据格式**: 在`data/`目录修改数据加载器

## 引用和致谢

如果使用本项目，请引用相关论文：
- PointNet++: Deep Hierarchical Feature Learning on Point Sets
- CycleGAN: Unpaired Image-to-Image Translation
- 以及其他相关的点云处理和风格迁移工作

本项目整合了多个开源库和研究成果，感谢原作者的贡献。