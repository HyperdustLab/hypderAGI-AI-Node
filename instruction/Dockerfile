FROM nvidia/cuda:12.5.0-runtime-ubuntu20.04

WORKDIR /app

# Install dependencies including git, gcc, and g++
RUN apt-get update && \
    apt-get install -y wget bzip2 git gcc g++ && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"

RUN conda init bash

# Create a conda environment and install Python 3.10
RUN conda create --name unsloth_env python=3.10 -y && \
    echo "conda activate unsloth_env" >> ~/.bashrc

# Ensure conda environment is activated and upgrade pip
SHELL ["conda", "run", "-n", "unsloth_env", "/bin/bash", "-c"]

RUN pip install --upgrade pip

# Copy the requirements.txt file
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt && echo "Installed dependencies" || { echo 'Failed to install dependencies'; exit 1; }

RUN mkdir -p \
    /models/unsloth_Meta-Llama-3.1-8B-bnb-4bit \
    /models//HyperdustProtocol/HyperAuto-cog-llama3-8b-3407 && \
    chmod -R 777 /models

# 使用huggingface-cli执行模型下载
RUN huggingface-cli download unsloth/Meta-Llama-3.1-8B-bnb-4bit \
    --local-dir /models/unsloth_Meta-Llama-3.1-8B-bnb-4bit \
    --resume-download \
    --local-dir-use-symlinks False \ 
    --cache-dir /tmp/hf_cache && \
    huggingface-cli download HyperdustProtocol/HyperAuto-cog-llama3-8b-3407 \
    --local-dir /models/HyperdustProtocol/HyperAuto-cog-llama3-8b-3407 \
    --resume-download \
    --local-dir-use-symlinks False \
    --cache-dir /tmp/hf_cache


COPY . .

CMD ["sh", "-c", "conda run --no-capture-output -n unsloth_env python3 /app/app.py"]
