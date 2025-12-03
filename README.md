# skeleton_cpp

一个基于 CMake 构建的 C++ 项目骨架，旨在提供模块化、可复用的 C++ 库开发模板。

## 项目结构

```
skeleton_cpp/
├── CMakeLists.txt                 # 根CMakeLists.txt，配置整个项目
├── README.md
├── include/                       # 公共头文件
│   └── skeleton_cpp/             # 项目名作为子目录
│       └── calculator/
│           └── Calculator.hpp    # Calculator类声明
├── src/                           # 源代码
│   ├── CMakeLists.txt            # 源码构建配置
│   └── calculator/
│       └── Calculator.cpp        # Calculator类实现
├── tests/                         # 测试代码
│   ├── CMakeLists.txt
│   └── calculator_test.cpp       # Calculator类测试
└── third_party/                   # 第三方依赖
```

## 编译步骤

### 标准构建

```bash
mkdir build && cd build
cmake ..
make
```

### 运行测试

```bash
./tests/calculator_test
```

或者使用CTest运行所有测试:

```bash
ctest
```

## 使用 find_package 的方式（推荐）

这是标准 CMake 生态中，使用包配置文件（package config）让其他项目「找到并使用你的库」。

编译步骤
在你的库项目中，配置并安装 package config 文件：

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
make install
```

这会把库文件、头文件和 cmake 配置文件安装到指定路径。

## 其他项目使用

```cmake
list(APPEND CMAKE_PREFIX_PATH "/path/to/install")  # 告诉 CMake 去哪里找包

find_package(skeleton_cpp REQUIRED)

target_link_libraries(myExecutable PRIVATE skeleton_cpp::skeleton_cpp)
```

这样你的库 skeleton_cpp 就像 OpenCV、Boost 一样被其他模块方便调用，且自动处理依赖和头文件路径。