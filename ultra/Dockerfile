FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

#------------------------------------------------------------------------------
# Certificate Setup
#------------------------------------------------------------------------------
#RUN apt-get update && apt-get install -y openssh-server
RUN apt-get install ca-certificates
RUN update-ca-certificates
RUN apt-get install apt-transport-https

#------------------------------------------------------------------------------
# Set timezone
#------------------------------------------------------------------------------
# Prevent tzdata from trying to be interactive
# ENV TZ=Europe/Minsk
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | \
          cat > /etc/apt/apt.conf.d/10no--check-valid-until

#------------------------------------------------------------------------------
# Nvidia Drivers Installation
#------------------------------------------------------------------------------
#ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display
#RUN apt-get update && apt-get install --no-install-recommends -y mesa-utils \
#    && rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------------------------
# Install dependencies
#------------------------------------------------------------------------------
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

RUN echo "Installing dependencies"
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

# Install pip dependencies
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3.7 get-pip.py && \
    pip install --upgrade pip

#------------------------------------------------------------------------------
# Display
#------------------------------------------------------------------------------
RUN echo "Installing XDummy"
ENV DISPLAY :1

RUN apt-get install -y \
    xserver-xorg-video-dummy \
    x11-apps

# VOLUME /tmp/.X11-unix
RUN wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf

#------------------------------------------------------------------------------
# Simulator
#------------------------------------------------------------------------------
RUN echo "Setup SMARTS Dependencies"
ENV SUMO_HOME /usr/share/sumo

#------------------------------------------------------------------------------
# Fix Git
#------------------------------------------------------------------------------
RUN cp /etc/apt/sources.list /etc/apt/sources.list~ && \
    sed -i 's/# deb-src/deb-src/' /etc/apt/sources.list && \
    apt-get update

RUN apt-get install -y \
    git

RUN apt-get install build-essential fakeroot dpkg-dev -y && \
    apt-get build-dep git -y && \
    apt-get install libcurl4-openssl-dev -y  && \
    cd ~ && \
    mkdir source-git && \
    cd source-git/ && \
    apt-get source git && \
    cd git-2.*.*/ && \
    sed -i 's/libcurl4-gnutls-dev/libcurl4-openssl-dev/' ./debian/control && \
    sed -i '/TEST\s*=\s*test/d' ./debian/rules && \
    dpkg-buildpackage -rfakeroot -b -uc -us && \
    dpkg -i ../git_*ubuntu*.deb

#------------------------------------------------------------------------------
# Clean-up
#------------------------------------------------------------------------------
RUN echo "Cleaning-up"
RUN apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------------------------
# Copy src files
#------------------------------------------------------------------------------
# Install SMARTS
RUN mkdir -p /SMARTS
#COPY ./setup.py /SMARTS/

# Add https://github.com/krallin/tini
ARG TINI_VERSION=v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh


#------------------------------------------------------------------------------
# Entrypoint
#------------------------------------------------------------------------------
# For Envision
EXPOSE 8081

# TODO: Find a better place to put this (e.g. through systemd). As it stands now ctrl-c
#       could close x-server. Even though it's "running in the background".
ENTRYPOINT ["/tini", "--", "entrypoint.sh"]

CMD ["/bin/bash"]
# Once in container can then run,
#   python3 examples/competition
