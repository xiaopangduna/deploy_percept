# Dev Containers

每个子目录对应一种开发容器配置，无根目录默认入口；打开容器时需显式选择配置。

## 目录结构

```
.devcontainer/
├── README.md
├── ubuntu-22.04-allwinner/    # Allwinner / Orange Pi（A733 等）交叉编译
│   ├── devcontainer.json
│   ├── Dockerfile
│   ├── post-create.sh
│   ├── post-start.sh
│   └── .dockerignore
└── ubuntu-20.04-sigmastar/    # Sigmastar / CoMake D1 交叉编译
    ├── devcontainer.json
    ├── Dockerfile
    ├── post-create.sh
    ├── post-start.sh
    └── .dockerignore
```

## 使用方式

1. 命令面板：`Dev Containers: Open Container Configuration File`
2. 选择目标配置，例如：
   - `.devcontainer/ubuntu-22.04-allwinner/devcontainer.json`
   - `.devcontainer/ubuntu-20.04-sigmastar/devcontainer.json`
3. **Rebuild and Reopen in Container**（修改 Dockerfile / devcontainer 配置后必须 Rebuild）

## ubuntu-22.04-allwinner

- 基础镜像：Ubuntu 22.04（orangepi-build 官方推荐宿主机版本）
- 运行用户：`vscode`（非 root；`updateRemoteUserUID` 与 WSL 宿主机 UID 对齐，避免 bind mount 出现 `nobody` 权限问题）
- 工作区：挂载 `deploy_percept` 上两级目录（`deep_learning`）到 `/workspace/deep_learning`
- 工具链：宿主机 `/opt/toolchains/` → 容器 `/opt/toolchains/`
- 预期工具链：`gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu`（A733 交叉编译）

工具链准备：

```bash
# 解压或链接到 /opt/toolchains/
ln -sfn /path/to/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu \
  /opt/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu
```

下载参考：https://mirrors.tuna.tsinghua.edu.cn/armbian-releases/_toolchain/

## ubuntu-20.04-sigmastar

- 基础镜像：`comake_d1_docker:latest`（需事先 `docker load`；Dockerfile 在其上叠一层，便于后续加装工具）
- apt 源：构建时将 `archive.ubuntu.com` / `security.ubuntu.com` 替换为清华镜像（与 allwinner 一致）
- 构建网络：`build.options: ["--network=host"]`（宿主机开了 Clash TUN/fake-ip 时，默认 bridge 会解析到 `198.18.x.x` 导致 apt 失败）
- 运行用户：`vscode`（`common-utils` + `updateRemoteUserUID`）
- 工作区：同 allwinner（`../..` → `/workspace/deep_learning`）
- 工具链：宿主机 `/opt/toolchains/` → 容器 `/opt/toolchains/`
- 预期工具链：`arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf`（前缀 `arm-unknown-linux-gnueabihf-`）
- sysroot：`.../arm-unknown-linux-gnueabihf/sysroot`

工具链准备：

```bash
# 解压 SDK 工具链到 /opt/toolchains/
# 例如来自 Pcupid DLC Linux_SDK Tools/ToolChain
tar -C /opt/toolchains -xf arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf.tar.xz
```

容器内验证：

```bash
arm-unknown-linux-gnueabihf-gcc --version
```

后续若需加装宿主机工具，编辑该目录 `Dockerfile` 中「可扩展工具层」后 Rebuild。
