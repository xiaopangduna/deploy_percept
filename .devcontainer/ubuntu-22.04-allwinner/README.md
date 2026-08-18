# ubuntu-22.04-allwinner

Allwinner / Orange Pi（A733 等）交叉编译环境。共用约定见 [../README.md](../README.md)。

## 环境

- 资料下载：[Orange Pi 4 Pro 服务与支持](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-4-Pro.html)（用户手册、Linux SDK 等）
- 无官方 Docker 镜像；本目录 Dockerfile 基于 `ubuntu:22.04`（orangepi-build 官方推荐宿主机版本），参考用户手册-4. Linux SDK——orangepi-build 使用说明
- 工具链说明：官方用户手册 Linux SDK「**4.1 orangepi-build 使用说明**」
- 本容器工具链：`gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu`（前缀 `aarch64-none-linux-gnu-`，Orange Pi 4 Pro / A733）
- CMake：系统包（22.04 自带 ≥3.22，支持 `cmake --preset`）

## 前置

无需 `docker load`。Rebuild 时拉取 `ubuntu:22.04`（本机已有则可跳过）。日常开发走本目录 Dev Container Rebuild（见 [../README.md](../README.md)）。

## 工具链准备

orangepi-build 会下载多套工具链；本容器只用 **aarch64** 这一套：

`gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu`

在开发板上确认架构（SSH 登录后）：

```bash
uname -a
```

Orange Pi 4 Pro 末尾应为 `aarch64`。

宿主机放到 `/opt/toolchains/`（devcontainer 会 bind mount 进容器）。任选一种：

**方式 A：清华镜像下载后解压**

```bash
sudo mkdir -p /opt/toolchains
# 包名见 https://mirrors.tuna.tsinghua.edu.cn/armbian-releases/_toolchain/
wget -P /tmp \
  https://mirrors.tuna.tsinghua.edu.cn/armbian-releases/_toolchain/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu.tar.xz
tar -C /opt/toolchains -xf \
  /tmp/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu.tar.xz
```

**方式 B：已有 orangepi-build 目录则做符号链接**

第一次跑 `./build.sh` 时会把工具链下到 `orangepi-build/toolchains/`：

```bash
sudo mkdir -p /opt/toolchains
ln -sfn /path/to/orangepi-build/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu \
  /opt/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu
```

完成后应存在：

```text
/opt/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/
```

## 构建后验证

Rebuild 完成后在容器内检查：

```bash
# 工作区
pwd
# 期望：/workspace/deep_learning/src/deploy_percept

# CMake
cmake --version
# 期望：≥ 3.22

# 交叉编译器（bind mount 后 PATH 由 post-create 注入）
aarch64-none-linux-gnu-gcc --version

# 工具链目录
ls /opt/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc
```

`post-create.sh` 在首次创建时也会探测工具链；未挂载时只会告警，不会失败。

## 构建注意

- apt 源：Dockerfile 将 `archive.ubuntu.com` / `security.ubuntu.com` 换为清华镜像，否则国内 `apt-get` 常失败。
- 加装工具：编辑本目录 `Dockerfile` 后 Rebuild。
