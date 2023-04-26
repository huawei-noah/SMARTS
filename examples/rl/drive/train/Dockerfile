FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install libraries.
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        git \
        libspatialindex-dev \
        python3.8 \
        python3.8-venv \
        xorg && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Update default python version.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Setup virtual environment and install pip.
ENV VIRTUAL_ENV=/opt/.venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip wheel

# Install requirements.txt .
COPY ./examples/rl/baseline/train/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source files and install.
COPY . /SMARTS
WORKDIR /SMARTS
RUN pip install -e .[camera_obs,argoverse]
RUN pip install -e ./examples/rl/drive/inference

SHELL ["/bin/bash", "-c", "-l"]