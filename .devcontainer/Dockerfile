# LTS (支持至 2030 年)
# FROM ubuntu:20.04 
# LTS (支持至 2032 年)  
FROM ubuntu:22.04  
# LTS (支持至 2034 年)
# FROM ubuntu:24.04     

# 避免在非交互模式下出现tzdata错误
ARG DEBIAN_FRONTEND=noninteractive

# 换 apt 源
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
 && sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list


RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gdb \
    cmake \
    ca-certificates \   
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* 

    


