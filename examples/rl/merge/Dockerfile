FROM tensorflow/tensorflow:2.4.0

ARG DEBIAN_FRONTEND=noninteractive

# (Optional) Disable certificate verification
RUN touch /etc/apt/apt.conf.d/99verify-peer.conf && \
    echo >>/etc/apt/apt.conf.d/99verify-peer.conf 'Acquire::https::Verify-Peer "false";' && \
    echo >>/etc/apt/apt.conf.d/99verify-peer.conf 'Acquire::https::Verify-Host "false";'

# (Temporary) NVIDIA invalid public key
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install libraries.
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        ffmpeg \
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
COPY ./examples/rl/merge/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# (Optional) Disable git certificate verification
RUN git config --global http.sslVerify false

# Copy source files and install.
COPY . /src
WORKDIR /src
RUN pip install --no-cache-dir -e ./examples/rl/merge

# For Envision.
EXPOSE 8081

# Optimizing compiler for machine learning. See https://www.tensorflow.org/xla .
ENV TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"

# Suppress message of missing /dev/input folder.
RUN echo "mkdir -p /dev/input" >> ~/.bashrc

SHELL ["/bin/bash", "-c", "-l"]