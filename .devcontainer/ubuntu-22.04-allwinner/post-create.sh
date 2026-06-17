#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="ubuntu-22.04-allwinner"
TOOLCHAIN_DIR="/opt/toolchains"
TOOLCHAIN_NAME="gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu"
TOOLCHAIN_ROOT="${TOOLCHAIN_DIR}/${TOOLCHAIN_NAME}"
PROFILE_SNIPPET="/etc/profile.d/allwinner-toolchain.sh"

echo "[${CONTAINER_NAME} post-create] Setting environment..."

cat > "${PROFILE_SNIPPET}" <<EOF
# Allwinner A733 交叉工具链（orangepi-build 推荐版本）
export TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT}"
export CROSS_COMPILE="aarch64-none-linux-gnu-"
if [ -d "\${TOOLCHAIN_ROOT}/bin" ]; then
    export PATH="\${TOOLCHAIN_ROOT}/bin:\${PATH}"
fi
EOF

if [ -d "${TOOLCHAIN_ROOT}/bin" ]; then
    echo "[${CONTAINER_NAME} post-create] Found toolchain: ${TOOLCHAIN_ROOT}"
    "${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-gcc" --version | head -1
else
    echo "[${CONTAINER_NAME} post-create] WARN: toolchain not found at ${TOOLCHAIN_ROOT}"
    echo "  Place ${TOOLCHAIN_NAME} under tmp/toolchains/ on the host, then rebuild or restart the container."
    echo "  Or run orangepi-build once to download it into tmp/orangepi-build/toolchains/ and symlink/copy here."
fi

echo "[${CONTAINER_NAME} post-create] Done."
