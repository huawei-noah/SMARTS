# Steps to build and push the SMARTS docker image
# ```bash
# $ cd </path/to/SMARTS>
# export VERSION=v2.0.1
# $ docker build --no-cache -f ./utils/docker/Dockerfile -t huaweinoah/smarts:$VERSION .
# $ docker login
# $ docker push huaweinoah/smarts:$VERSION
# ```
FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

# Prevent tzdata from trying to be interactive
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# Install libraries.
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        git \
        libenchant-dev \
        libspatialindex-dev \
        python3.8 \
        python3.8-venv \
        xorg && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Update default python version.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Setup python virtual environment and install pip.
ENV VIRTUAL_ENV=/opt/.venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip && pip install wheel==0.38.4

# Install requirements.txt .
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source files and install.
COPY . /src
WORKDIR /src
RUN pip install --no-cache-dir -e .[camera-obs,dev,rllib,test,torch,train] && \
    cp -r /src/smarts.egg-info /media/smarts.egg-info

# For Envision.
EXPOSE 8081
# For centralized TraCI management server
EXPOSE 8097

# Suppress message of missing /dev/input folder and copy smarts.egg-info if not there
RUN echo "mkdir -p /dev/input\n" \
         "if [[ ! -d /src/smarts.egg-info ]]; then" \
         "   cp -r /media/smarts.egg-info /src/smarts.egg-info;" \
         "   chmod -R 777 /src/smarts.egg-info;" \
         "fi" >> ~/.bashrc

SHELL ["/bin/bash", "-c", "-l"]
