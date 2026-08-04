# Sigmastar / CoMake D1 交叉编译环境
# 基镜像需事先 docker load：comake_d1_docker:latest
FROM comake_d1_docker:latest

ARG DEBIAN_FRONTEND=noninteractive

# 厂商镜像默认 archive.ubuntu.com，国内常不可达 → Unable to locate package
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
 && sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
 && apt-get update \
 && rm -rf /var/lib/apt/lists/*

# --- 可扩展工具层：后续在此追加 apt 包 ---
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cmake \
#     ninja-build \
#     && rm -rf /var/lib/apt/lists/*

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
