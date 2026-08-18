# ubuntu-20.04-sigmastar

Sigmastar / CoMake D1 交叉编译环境。共用约定见 [../README.md](../README.md)。

## 环境

- 下载（同一页，两个包）：[CoMake SDK 支持页](https://www.comake.online/support/sdk/)
  - **CoMake D1 SDK Docker**：容器镜像 `comake_d1_docker.tar.gz` 及说明
  - **Comake_D1_SDK**：Linux SDK（含工具链）及说明
- 基础镜像：`comake_d1_docker:latest`（需事先 `docker load`；Dockerfile 在其上叠一层）
- 本容器工具链：`arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf`（前缀 `arm-unknown-linux-gnueabihf-`，CoMake Pi D1 / SSD2355）
- sysroot：`.../arm-unknown-linux-gnueabihf/sysroot`
- CMake：Kitware 源安装（≥3.21，支持 `cmake --preset`；勿用系统自带 3.16）

## 前置

### 1. 确认是否已导入

```bash
docker images | grep comake_d1_docker
```

已有 `comake_d1_docker:latest` 且完好则可跳过第 2–3 步；厂商换包后再重新 load。

### 2. 拿到文件后不要当普通压缩包解压

`comake_d1_docker.tar.gz` 是 `docker save` 导出的 OCI 镜像包，**不是**源码/工具链压缩包。不要 `tar -xzf` 或右键解压；直接进入第 3 步 `docker load`。

### 3. 导入镜像

在宿主机执行（路径按实际位置改；约 1.3GB，需几分钟）：

```bash
# 推荐：不解压，直接 load
docker load -i tmp/comake_d1_docker.tar.gz
```

再确认：

```bash
docker images | grep comake_d1_docker
# 期望：comake_d1_docker    latest    ...    约 3.75GB
```

此镜像仅作基镜像；日常开发走本目录 Dev Container Rebuild（见 [../README.md](../README.md)），不要直接 `docker run comake_d1_docker`。

## 工具链准备

SDK 解压后工具链在：

```text
Pcupid_DLC173V2.3.3_0918/Linux_SDK_Release_20250918/Tools/ToolChain/
```

按**开发板架构**选包（本容器用第一条）：

| 压缩包 | 架构 | 用途 |
| --- | --- | --- |
| `arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf.tar.xz` | armv7 / gnueabihf | CoMake Pi D1（SSD2355），本环境 |
| `aarch64-unknown-linux-gcc-12.4.0-glibc-2.37-gnu.tar.xz` | aarch64 | 同 SDK 的 64 位板，不用于本容器 |
| `riscv_gcc.tar.xz` | RISC-V | 同 SDK 的 RISC-V 板，不用于本容器 |

在开发板上确认架构（SSH 登录后）：

```bash
uname -a
```

CoMake Pi D1 示例：

```text
Linux (none) 6.1.111-rt42 #2 SMP PREEMPT Wed Mar 11 14:10:04 CST 2026 armv7l GNU/Linux
```

末尾 `armv7l` 对应上表 gnueabihf 包。本容器选用：

`arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf`

宿主机解压到 `/opt/toolchains/`（devcontainer 会 bind mount 进容器）：

```bash
sudo mkdir -p /opt/toolchains
tar -C /opt/toolchains -xf \
  tmp/Pcupid_DLC173V2.3.3_0918/Linux_SDK_Release_20250918/Tools/ToolChain/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf.tar.xz
```

路径按 SDK 实际位置调整。解压后应存在：

```text
/opt/toolchains/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf/
```

## 构建后验证

Rebuild 完成后在容器内检查：

```bash
# 工作区
pwd
# 期望：/workspace/deep_learning/src/deploy_percept

# CMake（系统 3.16 会把 --preset 当成源码路径）
cmake --version
# 期望：≥ 3.21

# 交叉编译器
arm-unknown-linux-gnueabihf-gcc --version

# 工具链与 sysroot
ls /opt/toolchains/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf/bin/arm-unknown-linux-gnueabihf-gcc
ls /opt/toolchains/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf/arm-unknown-linux-gnueabihf/sysroot
```

`post-create.sh` 在首次创建时也会探测工具链；未挂载时只会告警，不会失败。

## 构建注意

- apt 源：Dockerfile 将 `archive.ubuntu.com` / `security.ubuntu.com` 换为清华镜像，否则国内 `apt-get` 常失败。
- 构建网络：`devcontainer.json` 中 `build.options: ["--network=host"]`。宿主机开了 Clash TUN/fake-ip 时，默认 bridge 会解析到 `198.18.x.x` 导致 apt 失败。
- 加装工具：编辑本目录 `Dockerfile` 后 Rebuild。
