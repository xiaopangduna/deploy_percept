# yolov5_detect_awnn

全志 VIPLite YOLOv5 检测（Orange Pi 4 Pro A733）。推理经 `deploy_percept::engine::AwnnEngine`。

## 资源（install → share/percept/apps/yolov5_detect_awnn/）

| 文件 | 说明 |
|------|------|
| `yolov5.nb` | A733 模型 |
| `dog.jpg` | 默认输入图 |

## 构建与安装

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release -DBUILD_AWNN_APPS=ON
cmake --build --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release --target yolov5_detect_awnn
bash scripts/install.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
```

## 运行

**install 树（板端）：**

```bash
cd ~/deploy_percept
export PERCEPT_ROOT=$PWD/share/percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./bin/yolov5_detect_awnn
```

默认输出：`share/percept/apps/yolov5_detect_awnn/yolov5_detect_awnn_out.jpg`
