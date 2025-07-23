FROM nvidia/cuda:12.5.0-cudnn9-devel-ubuntu24.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 设置工作目录
WORKDIR /workspace

# 更新系统并安装基础依赖
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libxi6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建python符号链接
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# 升级pip
RUN python -m pip install --upgrade pip setuptools wheel

# 安装PyTorch和CUDA 12.5兼容版本
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 复制requirements文件
COPY requirements.txt /workspace/requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装额外的依赖
RUN pip install \
    tensorboard \
    wandb \
    pytest \
    black \
    flake8 \
    ipython \
    jupyter

# 创建必要的目录
RUN mkdir -p /workspace/datasets \
    /workspace/checkpoints \
    /workspace/logs \
    /workspace/experiments \
    /workspace/results

# 设置用户权限（避免权限问题）
RUN useradd -m -u 1000 researcher && \
    chown -R researcher:researcher /workspace

# 切换到普通用户
USER researcher

# 暴露端口
EXPOSE 6006 8888

# 设置启动命令
CMD ["/bin/bash"]


# docker/docker-compose.yml
version: '3.8'

services:
  pointcloud-style-transfer:
    build:
      context: ../
      dockerfile: docker/Dockerfile
    image: pointcloud-style-transfer:latest
    container_name: pointcloud-style-transfer
    
    # GPU支持
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    # 环境变量
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量调整
      - PYTHONPATH=/workspace
      - WANDB_API_KEY=${WANDB_API_KEY}  # 如果使用wandb
      
    # 卷挂载
    volumes:
      # 代码目录
      - ../:/workspace
      # 数据集（可以挂载到外部存储）
      - ${DATA_DIR:-./datasets}:/workspace/datasets
      # 检查点（持久化保存）
      - ${CHECKPOINT_DIR:-./checkpoints}:/workspace/checkpoints
      # 日志
      - ${LOG_DIR:-./logs}:/workspace/logs
      # 实验结果
      - ${EXPERIMENT_DIR:-./experiments}:/workspace/experiments
      
    # 网络设置
    networks:
      - pcst-network
    
    # 端口映射
    ports:
      - "6006:6006"  # TensorBoard
      - "8888:8888"  # Jupyter
      
    # 资源限制
    shm_size: '16gb'  # 共享内存大小
    
    # 保持容器运行
    stdin_open: true
    tty: true
    
    # 健康检查
    healthcheck:
      test: ["CMD", "nvidia-smi"]
      interval: 30s
      timeout: 10s
      retries: 3
    
    # 自动重启
    restart: unless-stopped

  # TensorBoard服务（可选）
  tensorboard:
    image: tensorflow/tensorflow:latest-gpu
    container_name: pointcloud-tensorboard
    command: tensorboard --logdir=/logs --host=0.0.0.0
    volumes:
      - ${LOG_DIR:-./logs}:/logs:ro
    ports:
      - "6007:6006"
    networks:
      - pcst-network
    restart: unless-stopped

  # Jupyter服务（可选）
  jupyter:
    build:
      context: ../
      dockerfile: docker/Dockerfile
    image: pointcloud-style-transfer:latest
    container_name: pointcloud-jupyter
    command: jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - PYTHONPATH=/workspace
      
    volumes:
      - ../:/workspace
      - ${DATA_DIR:-./datasets}:/workspace/datasets:ro
      - ${CHECKPOINT_DIR:-./checkpoints}:/workspace/checkpoints:ro
      
    ports:
      - "8889:8888"
    
    networks:
      - pcst-network
      
    restart: unless-stopped

networks:
  pcst-network:
    driver: bridge

