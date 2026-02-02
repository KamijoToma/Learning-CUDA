ARG VARIANT="13.1.0-devel-ubuntu22.04"
FROM nvidia/cuda:${VARIANT}

ENV DEBIAN_FRONTEND=noninteractive

# 安装基础工具、LLVM/Clang 工具链、clangd 和 bear (用于生成编译数据库)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg \
    lsb-release \
    bear \
    build-essential \
    git \
    python3 \
    python3-pip \
    expect \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-18 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-18 \
        clangd-18 \
        llvm-18 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv: the package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}