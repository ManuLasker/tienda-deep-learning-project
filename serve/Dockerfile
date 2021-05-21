FROM ubuntu:latest

LABEL maintainer="emanuel.afanador@koombea.com"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

ARG PYTHON_VERSION=3.8

ENV DEBIAN_FRONTEND noninteractive

# System packages 
RUN apt-get update && apt-get install -y  apt-utils
RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    curl \
    gcc \
    mono-mcs \
    build-essential \
    ca-certificates \
    wget \
    pkg-config 

# Install miniconda and python
RUN curl -L -o ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda update conda \
    && /opt/conda/bin/conda install -y \
        python=$PYTHON_VERSION 

# Add conda to path
ENV PATH=/opt/conda/bin:${PATH}

# Copy requiremets to container
COPY config/environment /opt/ml/input/config/environment

# Install environment configuration
RUN pip install -U pip
RUN pip install --timeout=1000 -r /opt/ml/input/config/environment/requirements.txt
RUN pip install --timeout=1000 multi-model-server sagemaker-inference
RUN pip install --timeout=1000 torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --timeout=1000 opencv-python-headless

# Copy nginx config
COPY config/nginx.conf /opt/ml/input/config/nginx.conf

# Define importants env variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Copy project
COPY src /opt/ml/code

# Set our work dir
WORKDIR /opt/ml/code