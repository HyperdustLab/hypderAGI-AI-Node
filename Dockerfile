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

RUN conda create --name unsloth_env python=3.10 -y && \
    echo "conda activate unsloth_env" >> ~/.bashrc

SHELL ["conda", "run", "-n", "unsloth_env", "/bin/bash", "-c"]

RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" && echo "Installed unsloth" || { echo 'Failed to install unsloth'; exit 1; }
RUN pip install --no-deps trl==0.9.4 && echo "Installed trl" || { echo 'Failed to install trl'; exit 1; }
RUN pip install peft==0.11.1 && echo "Installed peft" || { echo 'Failed to install peft'; exit 1; }
RUN pip install accelerate==0.31.0 && echo "Installed accelerate" || { echo 'Failed to install accelerate'; exit 1; }
RUN pip install bitsandbytes==0.43.1 && echo "Installed bitsandbytes" || { echo 'Failed to install bitsandbytes'; exit 1; }
RUN pip install flask==3.0.3 && echo "Installed flask" || { echo 'Failed to install flask'; exit 1; }
RUN pip install nacos-sdk-python==0.1.14 && echo "Installed nacos-sdk-python" || { echo 'Failed to install nacos-sdk-python'; exit 1; }
RUN pip install eth-utils==4.1.1 && echo "Installed eth-utils" || { echo 'Failed to install eth-utils'; exit 1; }
RUN pip install xformers==0.0.26.post1 && echo "Installed xformers" || { echo 'Failed to install xformers'; exit 1; }

COPY . .

# Use an environment variable to specify the script to run
ENV SCRIPT_NAME="inference.py"

CMD ["sh", "-c", "conda run --no-capture-output -n unsloth_env python3 /app/${SCRIPT_NAME}"]
