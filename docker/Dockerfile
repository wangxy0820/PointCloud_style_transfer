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
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    # 图形和显示相关
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
    libfontconfig1-dev \
    # Python相关
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    # 开发工具
    vim \
    nano \
    htop \
    tree \
    zip \
    unzip \
    # 网络工具
    net-tools \
    iputils-ping \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建python符号链接
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# 升级pip和安装基础工具
RUN python -m pip install --upgrade pip setuptools wheel

# 安装PyTorch和CUDA 12.5兼容版本
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 复制requirements文件
COPY requirements.txt /workspace/requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装点云处理相关的额外依赖
RUN pip install \
    # 开发和调试工具
    tensorboard \
    wandb \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort \
    mypy \
    # Jupyter环境
    ipython \
    jupyter \
    jupyterlab \
    # 性能分析
    py-spy \
    memory-profiler \
    line-profiler

# 安装Open3D的系统依赖
RUN apt-get update && apt-get install -y \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建必要的目录
RUN mkdir -p /workspace/datasets \
    /workspace/checkpoints \
    /workspace/logs \
    /workspace/experiments \
    /workspace/results \
    /workspace/datasets/simulation \
    /workspace/datasets/real_world \
    /workspace/datasets/processed_hierarchical

# 设置用户权限（避免权限问题）
RUN useradd -m -u 1000 -s /bin/bash researcher && \
    chown -R researcher:researcher /workspace

# 切换到普通用户
USER researcher

# 暴露端口
EXPOSE 6006 8888 8501 5000 8080

# 设置启动命令
CMD ["/bin/bash"]