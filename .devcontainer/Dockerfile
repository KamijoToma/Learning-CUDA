ARG VARIANT="13.1.0-devel-ubuntu22.04"
FROM nvidia/cuda:${VARIANT}

ENV DEBIAN_FRONTEND=noninteractive

# 安装基础工具、clangd 和 bear (用于生成编译数据库)
RUN apt-get update && apt-get install -y --no-install-recommends \
    clangd \
    bear \
    build-essential \
    git \
    python3 \
    python3-pip \
    expect \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv: the package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}