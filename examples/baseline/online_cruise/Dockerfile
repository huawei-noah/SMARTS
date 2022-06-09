FROM tensorflow/tensorflow:2.4.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

ENV REQUESTS_CA_BUNDLE /etc/ssl/certs/ca-certificates.crt
ENV http_proxy=http://localhost:3128
ENV https_proxy=http://localhost:3128
ENV HTTP_PROXY=http://localhost:3128
ENV HTTPS_PROXY=http://localhost:3128
ADD certs /usr/local/share/ca-certificates/extra
RUN update-ca-certificates

# Install libraries.
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update --fix-missing 
RUN apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        libspatialindex-dev \
        python3.7 \
        python3.7-dev \
        python3.7-venv \
        xorg && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Update default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Setup virtual environment and install pip.
ENV VIRTUAL_ENV=/opt/.venv
RUN python3.7 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip wheel

# Setup SUMO.
RUN pip install eclipse-sumo==1.10.0
ENV SUMO_HOME $VIRTUAL_ENV/lib64/python3.7/site-packages/sumo

# Install requirements.txt .
COPY ./examples/sb3/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source files and install.
COPY . /src
WORKDIR /src
RUN pip install --no-cache-dir -e ./examples/baseline/online_cruise

# Build scenarios and maps
RUN scl scenario build-all --clean ./examples/baseline/online_cruise/scenarios

# For Envision
EXPOSE 8081

# Suppress message of missing /dev/input folder.
RUN echo "mkdir -p /dev/input" >> ~/.bashrc

SHELL ["/bin/bash", "-c", "-l"]
