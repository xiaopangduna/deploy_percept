# ubuntu-20.04-sigmastar

Sigmastar **通用**交叉编译 Dev Container（CoMake D1、SSC308QE 等共用）。共用约定见 [../README.md](../README.md)。

交叉工具链与容器解耦：容器只提供 OS / CMake / 联调工具，并挂载 `/opt/toolchains`；**选哪条链由** `cmake/toolchains/*.cmake` **与 CMake preset / `third_party_builder` 决定**，不在 Dockerfile 里写死 `TOOLCHAIN_ROOT`。

## 环境

- 基础镜像：`comake_d1_docker:latest`（需事先 `docker load`；Dockerfile 在其上叠一层）
- CMake：Kitware 源（≥3.21，支持 `cmake --preset`）
- 工具链目录：宿主机 `/opt/toolchains/` → 容器 `/opt/toolchains/`

| 平台 | cmake toolchain | `/opt/toolchains/` 下目录 | 来源 |
| --- | --- | --- | --- |
| CoMake Pi D1 (SSD2355) | `armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-toolchain.cmake` | `arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf` | CoMake 官网 SDK（见下） |
| SSC308QE (Iford) | `armv7l-linux-gnueabihf_ssc308qe-toolchain.cmake` | `gcc-11.1.0-20210608-sigmastar-glibc-x86_64_arm-linux-gnueabihf` | **供应商**发布包（见下） |

## 前置

### 1. 导入基镜像（CoMake Docker）

同一页下载两个包：[CoMake SDK 支持页](https://www.comake.online/support/sdk/)

- **CoMake D1 SDK Docker**：`comake_d1_docker.tar.gz`（容器基镜像）
- **Comake_D1_SDK**：Linux SDK（内含交叉工具链）

确认是否已导入：

```bash
docker images | grep comake_d1_docker
```

已有 `comake_d1_docker:latest` 且完好可跳过 load；厂商换包后再重新导入。

`comake_d1_docker.tar.gz` 是 `docker save` 导出的 OCI 镜像包，**不是**源码/工具链压缩包。不要 `tar -xzf` 或右键解压：

```bash
docker load -i <path>/comake_d1_docker.tar.gz
docker images | grep comake_d1_docker
# 期望：comake_d1_docker    latest    ...    约 3.75GB
```

此镜像仅作基镜像；日常开发走本目录 Dev Container Rebuild，不要直接 `docker run comake_d1_docker`。

### 2. CoMake Pi D1 交叉工具链（官网 SDK）

SDK 解压后工具链一般在：

```text
Pcupid_DLC173V2.3.3_*/Linux_SDK_Release_*/Tools/ToolChain/
```

按**开发板架构**选包（本工程 D1 用第一条）：

| 压缩包 | 架构 | 用途 |
| --- | --- | --- |
| `arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf.tar.xz` | armv7 / gnueabihf | **CoMake Pi D1（SSD2355），本工程使用** |
| `aarch64-unknown-linux-gcc-12.4.0-glibc-2.37-gnu.tar.xz` | aarch64 | 同 SDK 的 64 位板，不用于本 preset |
| `riscv_gcc.tar.xz` | RISC-V | 同 SDK 的 RISC-V 板，不用于本 preset |

在开发板上确认架构（SSH 登录后）：

```bash
uname -a
# CoMake Pi D1 示例末尾：armv7l  → 选 gnueabihf 包
```

宿主机解压到 `/opt/toolchains/`（devcontainer bind mount）：

```bash
sudo mkdir -p /opt/toolchains
tar -C /opt/toolchains -xf \
  <sdk>/Tools/ToolChain/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf.tar.xz
```

期望存在：

```text
/opt/toolchains/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf/bin/arm-unknown-linux-gnueabihf-gcc
```

### 3. SSC308QE 交叉工具链（供应商提供）

**不在 CoMake 官网下载**，由芯片/方案商随 Iford 发布包提供，例如：

```text
Iford_IMSSV05C13/Tools/ToolChain/gcc-11.1.0-20210608-sigmastar-glibc-x86_64_arm-linux-gnueabihf.tar.xz
```

（同目录还有 uclibc / baremetal 等；本工程默认用上述 **glibc 11.1.0** `arm-linux-gnueabihf`。）

解压：

```bash
sudo mkdir -p /opt/toolchains
tar -C /opt/toolchains -xf \
  <vendor>/Iford_IMSSV05C13/Tools/ToolChain/gcc-11.1.0-20210608-sigmastar-glibc-x86_64_arm-linux-gnueabihf.tar.xz
```

期望存在：

```text
/opt/toolchains/gcc-11.1.0-20210608-sigmastar-glibc-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc
```

可与 D1 工具链同时放在 `/opt/toolchains/`；构建时用不同 preset 选择。

## 构建后验证

```bash
pwd
# 期望：/workspace/deep_learning/src/deploy_percept

cmake --version
# 期望：≥ 3.21

ls /opt/toolchains

# 按平台选 preset（编译器绝对路径由 cmake toolchain 指定，无需容器 PATH）
cmake --preset armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release
# 或
cmake --preset armv7l-linux-gnueabihf_ssc308qe-release
```

## 构建注意

- apt 源：Dockerfile 将 `archive` / `security` 换为清华镜像。
- 构建网络：`build.options: ["--network=host"]`（Clash TUN/fake-ip 时默认 bridge 易失败）。
- 加装工具：编辑本目录 `Dockerfile` 后 Rebuild。
