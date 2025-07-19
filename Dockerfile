FROM nvidia/cuda:12.5.1-devel-ubuntu24.04

# 环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

# 系统包安装
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv \
    build-essential cmake git wget curl \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev \
    zlib1g-dev libjpeg-dev libpng-dev \
    libgl1-mesa-dev libegl1-mesa-dev libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev \
    libasound2t64 libx11-6 libxrandr2 libxi6 \
    && rm -rf /var/lib/apt/lists/*

# 修复Ubuntu 24.04 pip限制
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED
RUN mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "break-system-packages = true" >> /root/.pip/pip.conf

# Python符号链接
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 安装PyTorch (CUDA 12.5兼容)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 复制并安装Python依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed -r /tmp/requirements.txt

# 安装额外的点云和Jupyter包
RUN pip install torch-geometric jupyter jupyterlab ipywidgets

# 复制项目代码
COPY . /workspace/

# 创建必要目录
RUN mkdir -p /workspace/{datasets,checkpoints,logs,results,experiments}

# 设置Python路径
ENV PYTHONPATH=/workspace:$PYTHONPATH

# 暴露端口
EXPOSE 8888 6006 8080 5000 8050

# 启动命令
CMD ["/bin/bash"]
