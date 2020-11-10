FROM ubuntu:bionic

ARG DEBIAN_FRONTEND=noninteractive

# Prevent tzdata from trying to be interactive
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

RUN echo "Installing dependencies"
RUN apt-get update --fix-missing && \
    apt-get install -y \
    wget

RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y \
        wget \
        python3.7 \
        python3.7-dev \
        python3.7-venv \
        sumo \
        sumo-tools \
        sumo-doc \
        libspatialindex-dev \
        libsm6 \
        libxext6 \
        libxrender-dev

# Update default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Install pip dependencies
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python get-pip.py && \
    pip install --upgrade pip

RUN echo "Installing XDummy"
ENV DISPLAY :1

RUN apt-get install -y \
    xserver-xorg-video-dummy \
    x11-apps

# VOLUME /tmp/.X11-unix
RUN wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf
RUN cp /etc/X11/xorg.conf /usr/share/X11/xorg.conf.d/xorg.conf

RUN echo "Cleaning-up"
RUN apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN echo "Setup SMARTS Dependencies"
ENV SUMO_HOME /usr/share/sumo

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/src
COPY . /src
RUN cd /usr/local/lib/python3.7 && \
    mkdir site-packages && \
    cd site-packages && \
    python /src/setup.py develop
WORKDIR /src

# For Envision
EXPOSE 8081

# TODO: Find a better place to put this (e.g. through systemd). As it stands now ctrl-c
#       could close x-server. Even though it's "running in the background".
RUN echo "/usr/bin/Xorg " \
    "-noreset +extension GLX +extension RANDR +extension RENDER" \
    "-logfile ./xdummy.log -config /etc/X11/xorg.conf -novtswitch $DISPLAY &" >> ~/.bashrc

SHELL ["/bin/bash", "-c", "-l"]
