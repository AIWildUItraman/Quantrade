# 【生存容器镜像】TimesNet 虚拟货币交易系统
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    htop \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 升级 pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 设置 pip 源为清华


# 安装 PyTorch（CUDA 11.8 版本）
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 安装 Python 依赖
RUN pip3 install --no-cache-dir \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    tqdm==4.65.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    tensorboard==2.13.0 \
    jupyter==1.0.0

# 创建项目目录结构
RUN mkdir -p /workspace/crypto_timesnet/data \
    /workspace/crypto_timesnet/checkpoints \
    /workspace/crypto_timesnet/logs \
    /workspace/crypto_timesnet/layers

# 设置 Python 路径
ENV PYTHONPATH=/workspace/crypto_timesnet:$PYTHONPATH

# 验证 CUDA 环境
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); \
    print(f'CUDA Available: {torch.cuda.is_available()}'); \
    print(f'CUDA Version: {torch.version.cuda}'); \
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')"

# 默认命令
CMD ["/bin/bash"]
