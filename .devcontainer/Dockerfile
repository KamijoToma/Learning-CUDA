ARG VARIANT="13.1.0-devel-ubuntu22.04"
FROM nvidia/cuda:${VARIANT}

ENV DEBIAN_FRONTEND=noninteractive

# 替换为国内源（以清华源为例）
RUN sed -i.bak 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list \
    && sed -i.bak 's|http://security.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

# 配置时区等
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

# 安装基础工具、LLVM/Clang 工具链、clangd 和 bear (用于生成编译数据库)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg \
    lsb-release \
    bear \
    build-essential \
    git \
    openssh-client \
    curl \
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

RUN (curl -fsSL https://claude.ai/install.sh | bash) && echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Persist history in a mounted volume
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo "$SNIPPET" >> "/root/.bashrc"

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}