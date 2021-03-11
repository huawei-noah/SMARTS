# Steps to build and push minimal SMARTS docker image
# ```bash
# $ export VERSION=v0.4.13
# $ cd /path/to/SMARTS
# $ docker build -t huaweinoah/smarts:$VERSION-minimal -f minimal.Dockerfile .
# $ docker push huaweinoah/smarts:$VERSION-minimal
# ```

FROM ubuntu:bionic

# Install libraries
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y \
        libsm6 \
        libspatialindex-dev \
        libxext6 \
        libxrender-dev \
        python3.7 \
        python3.7-dev \
        python3.7-venv \
        sumo \
        sumo-doc \
        sumo-tools \
        wget \
        x11-apps \
        xserver-xorg-video-dummy && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Update default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python get-pip.py && \
    pip install --upgrade pip

# Setup DISPLAY
ENV DISPLAY :1
# VOLUME /tmp/.X11-unix
RUN wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf && \
    cp /etc/X11/xorg.conf /usr/share/X11/xorg.conf.d/xorg.conf

# Setup SUMO
ENV SUMO_HOME /usr/share/sumo

# For Envision
EXPOSE 8081

# TODO: Find a better place to put this (e.g. through systemd). As it stands now ctrl-c
#       could close x-server. Even though it's "running in the background".
RUN echo "/usr/bin/Xorg " \
    "-noreset +extension GLX +extension RANDR +extension RENDER" \
    "-logfile ./xdummy.log -config /etc/X11/xorg.conf -novtswitch $DISPLAY &" >> ~/.bashrc

SHELL ["/bin/bash", "-c", "-l"]
