## 基于Diffusion模型的LiDAR点云风格迁移
1. 项目概述
本项目旨在利用无监督的深度学习方法，实现两种不同域（例如 simulation 与 real-world）的LiDAR点云之间的风格迁移。其核心目标是，在严格保持源点云（如仿真数据）的精确几何结构（内容）的同时，为其赋予目标域点云（如真实世界数据）的独特风格特征，例如传感器噪声、扫描伪影和点密度分布等。

最终产出一个高保真的、融合了源内容与目标风格的新点云。

2. 核心挑战与解决方案
点云生成任务，特别是风格迁移，面临着巨大的挑战。模型需要同时学习两个相互冲突的目标：保持几何与转换风格。在训练初期，这种冲突极易导致梯度不稳定、损失爆炸，最终使模型输出崩溃，生成无意义的点云团（例如一个立方体或球体）。

为了从根本上解决这个问题，我们设计并实施了一套健壮的两阶段训练法 (Two-Stage Training)。

## 阶段一：几何重建预训练 (学习“画骨”)
目标：在此阶段，我们完全忽略风格迁移，专注于让模型成为一个完美的点云自动编码器。模型只学习一件事：输入一个点云，对其加噪，然后精确地重建出原始的、未加噪的点云。

策略：

自我重建任务：训练时，内容和风格都取自同一个源点云。

课程学习：在训练初期（前10个epoch），完全禁用所有几何损失（如Chamfer Loss），让模型只专注于学习基础的去噪任务。之后，再将几何损失的权重从0线性增加到目标值，引导模型平稳地学习几何约束。

产出：一个几何保持能力极强的预训练模型。其内容编码器（Content Encoder）已精通于提取点云的几何结构，而U-Net主干也学会了如何根据这些结构信息进行精确还原。

## 阶段二：风格迁移微调 (学习“画皮”)
目标：在第一阶段获得的强大几何保持能力的基础上，安全地引入并学习风格转换。

策略：

加载预训练权重：以第一阶段训练出的最佳模型作为起点。

冻结内容编码器：在第二阶段的训练中，Content Encoder 的权重被完全冻结，不允许更新。这就像一个坚固的“锚”，强制U-Net在学习新风格时不能以牺牲几何结构为代价。

引入风格损失：此时，我们可以放心地引入风格损失，并使用一个较低的学习率进行微调，让模型在不破坏“骨架”的前提下，巧妙地将风格“画”上去。

产出：一个最终模型，它既能精确地保持输入的内容，又能成功地渲染上目标的风格。

3. 项目结构

```
pointcloud_style_transfer/
├── config/
│   ├── __init__.py
│   └── config.py                  # 配置管理
├── models/
│   ├── __init__.py
│   ├── diffusion_model.py         # Diffusion模型核心
│   ├── pointnet2_encoder.py       # PointNet++特征提取
│   ├── chunk_fusion.py            # 块融合模块
│   └── losses.py                  # 损失函数定义
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # 数据集类
│   ├── preprocessing.py           # 数据预处理
│   └── augmentation.py            # 数据增强
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # 训练器
│   ├── progressive_trainer.py     # 渐进式训练
│   └── validator.py               # 验证器
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # 评估指标
│   └── tester.py                  # 测试器
├── utils/
│   ├── __init__.py
│   ├── visualization.py           # 可视化工具
│   ├── logger.py                  # 日志管理
│   └── checkpoint.py              # 检查点管理
├── scripts/
│   ├── preprocess_data.py         # 数据预处理脚本
│   ├── train.py                   # 训练脚本
│   ├── test.py                    # 测试脚本
│   ├── inference.py               # 推理脚本
│   └── visualize_results.py       # 结果可视化脚本
├── docker/
│   ├── Dockerfile                 # Docker镜像定义
│   ├── docker-compose.yml         # Docker Compose配置
│   └── requirements.txt           # Python依赖
├── datasets/                      # 数据目录
│   ├── simulation/               # 仿真点云
│   ├── real_world/              # 真实点云
│   └── processed/               # 预处理后的数据
├── experiments/                   # 实验结果
├── checkpoints/                   # 模型检查点
├── logs/                         # 训练日志
└── README.md                     # 项目说明

```

## 环境要求

- Ubuntu 24.04
- CUDA 12.5
- Python 3.10+
- PyTorch 2.1+
- 至少16GB GPU内存（推荐24GB+）

## 快速开始

### 1. 使用Docker（推荐）

```bash
# 克隆项目
git clone https://github.com/your-repo/pointcloud-style-transfer.git
cd pointcloud-style-transfer

# 构建并启动Docker容器
docker-compose up -d

# 进入容器
docker exec -it pointcloud-style-transfer bash
```

### 2. 本地安装

```bash
# 创建虚拟环境
conda create -n pc_style python=3.10
conda activate pc_style

# 安装PyTorch (CUDA 12.5)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

## 详细使用指南

### 步骤1: 数据准备

将您的点云数据组织成以下结构：
```
datasets/
├── simulation/
│   ├── sim_001.npy  # shape: (120000, 3)
│   ├── sim_002.npy
│   └── ...
└── real_world/
    ├── real_001.npy  # shape: (120000, 3)
    ├── real_002.npy
    └── ...
```

### 步骤2: 数据预处理

```bash
python scripts/preprocess_data.py \
    --sim_dir datasets/simulation \
    --real_dir datasets/real_world \
    --output_dir datasets/processed \
    --chunk_size 4096 \
    --overlap_ratio 0.2
```

参数说明：
- `--chunk_size`: 每个块的点数（默认2048）
- `--overlap_ratio`: 块之间的重叠率（默认0.3）
- `--num_workers`: 并行处理的进程数

### 步骤3: 训练模型

```bash
#supervised training
# python scripts/train.py \
#     --data_dir datasets/processed \
#     --experiment_name stage1 \
#     --batch_size 8 \
#     --num_epochs 40
    
#unsupervised training_stage1
python scripts/train_unsupervised.py

#unsupervised training_stage2
python scripts/train_unsupervised.py \
    --stage 2 \
    --stage1_checkpoint "experiments/stage1/checkpoints/best_model.pth"

### 步骤4: 测试模型

```bash
python scripts/test.py \
    --checkpoint experiments/my_experiment/checkpoints/best_model.pth \
    --test_data datasets/processed \
    --compute_all_metrics

#unsupervised testing
python scripts/test_unsupervised.py \
    --checkpoint experiments/test1/checkpoints/latest.pth \
    --test_data datasets/processed \
    --compute_all_metrics
```

### 步骤5: 推理（转换新的点云）

单个文件：
```bash
#supervised inference
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --sim_input path/to/simulation.npy \
    --real_reference path/to/reference.npy \
    --output path/to/output.npy

#unsupervised inference
python scripts/inference_unsupervised.py \
    --checkpoint experiments/stage2/checkpoints/best_model.pth \
    --sim_input path/to/simulation.npy \
    --real_reference path/to/reference.npy \
    --output path/to/output.npy
```

批量处理：
```bash
#supervised inference
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --sim_folder path/to/sim_folder \
    --real_reference path/to/real_reference.npy \
    --output_folder path/to/output_folder \
    --batch_process

#unsupervised inference
python scripts/inference_unsupervised.py \
    --checkpoint experiments/stage2/checkpoints/best_model.pth \
    --source datasets/test/000000.npy \
    --reference datasets/real_world/000000.npy \
    --output results/000000.npy
```

### 步骤6: 可视化结果

```bash
python scripts/visualize_results.py \
    --original datasets/simulation/000000.npy \
    --generated results/000000.npy \
    --reference datasets/real_world/000000.npy \
    --output_path visualization.png
```

## 配置参数详解

### 主要配置 (config/config.py)

```python
# 数据参数
total_points: 120000      # 完整点云点数
chunk_size: 4096         # 每个块的点数
overlap_ratio: 0.2       # 块重叠率

# 模型参数
model_type: "diffusion"  # 模型类型
num_timesteps: 1000      # Diffusion步数
beta_schedule: "cosine"  # 噪声调度

# stage1 训练参数
batch_size: 8            # 批大小
num_epochs: 100          # 训练轮数
learning_rate: 1e-5    # 学习率

# stage1 损失权重
lambda_diffusion: float = 1.0
lambda_chamfer: float = 10.0
lambda_content: float = 1.0
lambda_style: float = 0.0  # 在第一阶段完全禁用风格损失
lambda_lidar_structure: float = 1.0
lambda_smooth: float = 0.5

 # 第二阶段
batch_size: 8            
num_epochs: 100 
learning_rate: float = 1e-5

lambda_diffusion: float = 1.0
lambda_chamfer: float = 5.0
lambda_content: float = 1.0
lambda_style: float = 0.05
lambda_lidar_structure: float = 1.0
lambda_smooth: float = 0.5
```

## 训练技巧

1. **内存优化**：
   - 减小`batch_size`和`chunk_size`
   - 使用梯度累积
   - 启用混合精度训练

2. **训练稳定性**：
   - 使用EMA（指数移动平均）
   - 渐进式训练（从少量块开始）
   - 合适的学习率调度

3. **质量提升**：
   - 增大`overlap_ratio`提高块融合质量
   - 调整损失权重平衡各项指标
   - 使用更多的训练数据

## 常见问题

### Q1: 内存不足怎么办？

```bash
# 减小批大小和块大小
python scripts/train.py \
    --batch_size 4 \
    --chunk_size 1024 \
    --gradient_accumulation_steps 4
```

### Q2: 训练损失不下降？

- 检查数据预处理是否正确
- 尝试调整学习率
- 确保仿真和真实点云对应关系正确

### Q3: 生成结果有明显块边界？

- 增大`overlap_ratio`到0.4或0.5
- 增加`lambda_boundary`权重
- 使用更多训练轮数

## 性能基准

在NVIDIA A100 GPU上的测试结果：

| 指标 | 数值 |
|------|------|
| 训练速度 | ~50 batch/min |
| 推理速度 | ~2 秒/点云 |
| GPU内存使用 | ~12GB |
| 最终Chamfer距离 | 0.0015 |

## 扩展功能

### 1. 多GPU训练

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --data_dir datasets/processed \
    --distributed
```

### 2. 混合精度训练

```bash
python scripts/train.py \
    --data_dir datasets/processed \
    --use_amp \
    --amp_level O1
```

### 3. 实时监控

使用TensorBoard：
```bash
tensorboard --logdir experiments/my_experiment/logs
```

## 许可证

MIT License

## 引用

如果您使用本项目，请引用：
```bibtex
@misc{pointcloud_style_transfer,
  title={Point Cloud Style Transfer with Diffusion Models},
  author={WANG XINYU},
  year={2024},
  publisher={GitHub},
  url={https://github.com/wangxy0820/pointcloud-style-transfer}
}
```
