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
└── <future-container>/        # 后续新增容器
```

## 使用方式

1. 命令面板：`Dev Containers: Open Container Configuration File`
2. 选择 `.devcontainer/ubuntu-22.04-allwinner/devcontainer.json`
3. **Reopen in Container** / 重建容器

## ubuntu-22.04-allwinner

- 基础镜像：Ubuntu 22.04（orangepi-build 官方推荐宿主机版本）
- 挂载：`tmp/toolchains` → 容器内 `/opt/toolchains`
- 预期工具链：`gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu`（A733 内核 / 应用交叉编译）

工具链准备示例：

```bash
# 方式 1：orangepi-build 首次构建自动下载后，链接到 tmp/toolchains
ln -sfn "$(pwd)/tmp/orangepi-build/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu" \
  tmp/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu

# 方式 2：手动解压官方包到 tmp/toolchains/
```

## 新增容器

1. 复制 `ubuntu-22.04-allwinner/` 为模板
2. 修改 `Dockerfile`、`devcontainer.json` 中的 `name`、挂载与依赖
3. 通过 `Open Container Configuration File` 选择新配置
