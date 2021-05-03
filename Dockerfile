FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7


LABEL maintainer="emanuel.afanador@koombea.com"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

ARG PYTHON_VERSION=3.7

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

# Copy nginx config
COPY config/nginx.conf /opt/ml/input/config/nginx.conf

# Copy requiremets to container
# COPY config /opt/ml/input/config

# Define importants env variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Copy project
COPY src /opt/ml/code

# Set our work dir
WORKDIR /opt/ml/code