## 基于Diffusion模型的LiDAR点云风格迁移
项目结构

```
Style_transfer/
├── config/
│   ├── __init__.py
│   └── config.py                  # 配置管理
├── models/
│   ├── __init__.py
│   ├── diffusion_model.py         # Diffusion模型核心
│   ├── pointnet2_encoder.py       # PointNet++特征提取
│   └── losses.py                  # 损失函数定义
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # 数据集类
│   ├── preprocessing.py           # 数据预处理
│   └── augmentation.py            # 数据增强
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # 训练器
│   └── validator.py               # 验证器
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # 评估指标
│   └── tester.py                  # 测试器
├── utils/
│   ├── __init__.py
│   ├── visualization.py           # 可视化工具
│   ├── logger.py                  # 日志管理
│   ├── ema.py                     # EMA管理
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
│   ├── simulation/                # 仿真点云
│   ├── real_world/                # 真实点云
│   └── processed_hierarchical/    # 预处理后的数据
├── checkpoints/                   # 模型检查点
├── logs/                          # 训练日志
└── README.md                      # 项目说明

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
    --output_dir datasets/processed_hierarchical \
```

### 步骤3: 训练模型

```bash
#training
python scripts/train.py


### 步骤4: 测试模型

```bash
testing
python scripts/test.py \
    --checkpoint checkpoints/train/best_model.pth \
    --test_data datasets/processed_hierarchical \
    --compute_all_metrics
```

### 步骤5: 推理（转换新的点云）

单个文件：
```bash
inference
python scripts/inference.py \
    --checkpoint checkpoints/train/best_model.pth \
    --source datasets/test/000000.npy \
    --reference datasets/real_world/000000.npy \
    --output results/000000.npy
```

批量处理：
```bash
inference
python scripts/inference.py \
    --checkpoint checkpoints/train/best_model.pth \
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


### 实时监控

使用TensorBoard：
```bash
tensorboard --logdir logs/test2
```
