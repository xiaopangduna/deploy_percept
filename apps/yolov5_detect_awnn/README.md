# yolov5_detect_awnn

全志 VIPLite YOLOv5 检测（Orange Pi 4 Pro A733）。

当前：**VIPLite 链接验证**，不调用 NPU 推理。

## 依赖

NPU 运行时仅 **viplite-tina**（`AWNN::VIPLite`）。准备步骤见 `cmake/modules/FindAwnn.cmake`。

## 构建与安装

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release -DBUILD_AWNN_APPS=ON
cmake --build --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release --target yolov5_detect_awnn
bash scripts/install.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
```

install 打包 `lib/libNBGlinker.so`、`lib/libVIPhal.so`。
