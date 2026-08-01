# Allwinner / Orange Pi 交叉编译与 orangepi-build 宿主机环境
# LTS (支持至 2032 年)
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
 && sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# 基础构建工具 + orangepi-build / ai-sdk 常用宿主机依赖（精简子集）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gdb \
    cmake \
    ca-certificates \
    curl \
    wget \
    rsync \
    unzip \
    file \
    pkg-config \
    python3 \
    python3-dev \
    python3-distutils \
    qemu-user-static \
    device-tree-compiler \
    u-boot-tools \
    libncurses-dev \
    libssl-dev \
    zlib1g-dev \
    bison \
    flex \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    openssh-client \
    sshpass \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# 工具链挂载点（宿主机 tmp/toolchains -> /opt/toolchains）
RUN mkdir -p /opt/toolchains

# CI 用普通用户（ssh/rsync 需要 /etc/passwd 条目）。默认 1100，避免与
# devcontainer 的 vscode(1000) 冲突；CI 构建时传入 runner uid/gid：
#   --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)
ARG USERNAME=ci
ARG USER_UID=1100
ARG USER_GID=1100
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
 && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"
