#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="ubuntu-20.04-sigmastar"

echo "[${CONTAINER_NAME} post-create] Checking /opt/toolchains (bind mount)..."

if [ ! -d /opt/toolchains ]; then
    echo "[${CONTAINER_NAME} post-create] WARN: /opt/toolchains missing"
else
    # 只列顶层目录，不绑定某一平台；具体路径由 cmake/toolchains/*.cmake 指定
    echo "[${CONTAINER_NAME} post-create] Contents:"
    ls -1 /opt/toolchains | sed 's/^/  /' || true
fi

echo "[${CONTAINER_NAME} post-create] cmake $(cmake --version | head -1)"
echo "[${CONTAINER_NAME} post-create] Select platform via cmake --preset / third_party_builder (see cmake/toolchains/)."
echo "[${CONTAINER_NAME} post-create] Done."
