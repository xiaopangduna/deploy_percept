# Sigmastar / CoMake D1 交叉编译环境
# 基镜像需事先 docker load：comake_d1_docker:latest
FROM comake_d1_docker:latest

ARG DEBIAN_FRONTEND=noninteractive

# 厂商镜像默认 archive.ubuntu.com，国内常不可达 → Unable to locate package
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
 && sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
 && apt-get update \
 && rm -rf /var/lib/apt/lists/*

# CMake >= 3.21（CMakePresets.json version 3 / cmake --preset）；
# Ubuntu 20.04 自带 3.16 会把 --preset 当成源码路径。用 Kitware 源装新版。
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
 && curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
 && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
    > /etc/apt/sources.list.d/kitware.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends cmake \
 && cmake --version \
 && rm -rf /var/lib/apt/lists/*

# 板端联调辅助工具（rsync 已由基镜像 comake_d1_docker 提供，此处不重复安装）
# - openssh-client / sshpass：ssh/scp 推包与非交互登录
# - iputils-ping：网络连通性检查
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    sshpass \
    iputils-ping \
   #  rsync \ # 已由基镜像 comake_d1_docker 提供
 && rm -rf /var/lib/apt/lists/*

# 工具链挂载点（宿主机 /opt/toolchains -> 容器 /opt/toolchains）
RUN mkdir -p /opt/toolchains

# 交叉工具链环境（bind mount 后可用；未挂载时 PATH 仍无害）
ENV TOOLCHAIN_ROOT=/opt/toolchains/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf
ENV CROSS_COMPILE=arm-unknown-linux-gnueabihf-
ENV PATH=/opt/toolchains/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf/bin:${PATH}

RUN printf '%s\n' \
    '# Sigmastar / CoMake D1 交叉工具链（gcc 12.4 glibc 2.37 gnueabihf）' \
    'export TOOLCHAIN_ROOT=/opt/toolchains/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf' \
    'export CROSS_COMPILE=arm-unknown-linux-gnueabihf-' \
    'if [ -d "${TOOLCHAIN_ROOT}/bin" ]; then' \
    '    export PATH="${TOOLCHAIN_ROOT}/bin:${PATH}"' \
    'fi' \
    > /etc/profile.d/sigmastar-toolchain.sh \
 && chmod 644 /etc/profile.d/sigmastar-toolchain.sh

# CI 用普通用户（避免与 vscode(1000) 冲突；vscode 由 common-utils feature 创建）
ARG USERNAME=ci
ARG USER_UID=1100
ARG USER_GID=1100
RUN groupadd --gid "${USER_GID}" "${USERNAME}" 2>/dev/null || true \
 && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" 2>/dev/null || true
