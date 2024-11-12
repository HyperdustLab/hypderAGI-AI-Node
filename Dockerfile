FROM nvidia/cuda:12.5.0-runtime-ubuntu20.04

WORKDIR /app

# Install dependencies including git, gcc, and g++
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y wget bzip2 git gcc g++ && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"


ENV HTTP_PROXY="http://192.168.101.5:7890"
ENV HTTPS_PROXY="http://192.168.101.5:7890"
ENV NO_PROXY="localhost,127.0.0.1,.aliyuncs.com,.npmmirror.com"

RUN wget -q --spider --no-check-certificate https://www.google.com || { echo "Failed to access Google - proxy may not be working"; exit 1; }


RUN conda init bash

RUN conda create --name unsloth_env python=3.10 -y && \
    echo "conda activate unsloth_env" >> ~/.bashrc


RUN mkdir -p ~/.pip && echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.pip/pip.conf


SHELL ["conda", "run", "-n", "unsloth_env", "/bin/bash", "-c"]

COPY requirements.txt .

# 使用单个RUN命令安装所有依赖
RUN pip install --no-deps -r requirements.txt && \
    echo "All dependencies installed successfully" || { echo 'Failed to install dependencies'; exit 1; }

COPY . .

# Use an environment variable to specify the script to run
ENV SCRIPT_NAME="inference.py"

CMD ["sh", "-c", "conda run --no-capture-output -n unsloth_env python3 /app/${SCRIPT_NAME}"]
