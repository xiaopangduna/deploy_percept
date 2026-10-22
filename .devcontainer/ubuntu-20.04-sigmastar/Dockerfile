# Sigmastar 通用交叉编译 Dev Container（CoMake D1 / SSC308QE 等共用）
# 基镜像需事先 docker load：comake_d1_docker:latest
# 交叉工具链不在此写死：由 cmake/toolchains/*.cmake + CMake preset / third_party_builder 指定。
FROM comake_d1_docker:latest

ARG DEBIAN_FRONTEND=noninteractive

# 厂商镜像默认 archive.ubuntu.com，国内常不可达 → Unable to locate package
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
 && sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
 && apt-get update \
 && rm -rf /var/lib/apt/lists/*

# CMake >= 3.21（CMakePresets.json version 3 / cmake --preset）；
# Ubuntu 20.04 自带 3.16 会把 --preset 当成源码路径。用 Kitware 源装 3.31（钉死，避免 apt 跟到 4.x
# 后因 cmake_minimum_required < 3.5 导致旧第三方如 cnpy/OpenCV 配置失败）。
# 先 purge 系统 cmake：Kitware 把 cmake.m4 挪到 cmake 包，直接升级会与旧 cmake-data 文件冲突。
ARG CMAKE_KITWARE_VERSION=3.31.12-0kitware2ubuntu20.04.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
 && curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
 && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
    > /etc/apt/sources.list.d/kitware.list \
 && apt-get update \
 && apt-get remove -y --purge cmake cmake-data \
 && apt-get install -y --no-install-recommends \
    "cmake=${CMAKE_KITWARE_VERSION}" \
    "cmake-data=${CMAKE_KITWARE_VERSION}" \
 && apt-mark hold cmake cmake-data \
 && cmake --version \
 && rm -rf /var/lib/apt/lists/*

# 板端联调辅助工具（rsync 已由基镜像 comake_d1_docker 提供，此处不重复安装）
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    sshpass \
    iputils-ping \
 && rm -rf /var/lib/apt/lists/*

# 工具链挂载点（宿主机 /opt/toolchains -> 容器 /opt/toolchains）
# 各平台包按原名解压到 /opt/toolchains/<包名>/，具体路径见 cmake/toolchains/
RUN mkdir -p /opt/toolchains

# CI 用普通用户（避免与 vscode(1000) 冲突；vscode 由 common-utils feature 创建）
ARG USERNAME=ci
ARG USER_UID=1100
ARG USER_GID=1100
RUN groupadd --gid "${USER_GID}" "${USERNAME}" 2>/dev/null || true \
 && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" 2>/dev/null || true
