#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="ubuntu-20.04-sigmastar"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/opt/toolchains/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf}"

echo "[${CONTAINER_NAME} post-create] Checking toolchain..."

if [ -x "${TOOLCHAIN_ROOT}/bin/arm-unknown-linux-gnueabihf-gcc" ]; then
    echo "[${CONTAINER_NAME} post-create] Found toolchain: ${TOOLCHAIN_ROOT}"
    "${TOOLCHAIN_ROOT}/bin/arm-unknown-linux-gnueabihf-gcc" --version | head -1
else
    echo "[${CONTAINER_NAME} post-create] WARN: toolchain not found at ${TOOLCHAIN_ROOT}"
    echo "  Place arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf under /opt/toolchains/ on the host,"
    echo "  then rebuild or restart the container."
fi

echo "[${CONTAINER_NAME} post-create] Done."
