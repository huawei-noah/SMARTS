FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# FROM registry.kyber.team/kyber/lcc-team/cuda10.1-pytorch1.4-ubuntu18.04:latest

ARG DEBIAN_FRONTEND=noninteractive

#------------------------------------------------------------------------------
# Environment variables inside docker
#------------------------------------------------------------------------------
# Uncomment inside Huawei domain
ENV http_proxy http://localhost:3128
ENV HTTP_PROXY http://localhost:3128
ENV https_proxy http://localhost:3128
ENV HTTPS_PROXY http://localhost:3128
ENV ftp_proxy http://localhost:3128
ENV FTP_PROXY http://localhost:3128
ENV no_proxy 127.0.0.1,10.218.163.63,localhost,.huawei.com,.kyber.team
ENV NO_PROXY 127.0.0.1,10.218.163.63,localhost,.huawei.com,.kyber.team

ENV GIT_SSL_NO_VERIFY true

#------------------------------------------------------------------------------
# Certificate Setup
#------------------------------------------------------------------------------
#RUN apt-get update && apt-get install -y openssh-server
RUN apt-get install ca-certificates
COPY ultra/docker/certs/ /usr/local/share/ca-certificates/extra/
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
COPY ./setup.py /SMARTS/


RUN python3.7 -m venv /SMARTS/.venv
RUN bash -c 'echo -p python --version'
RUN /bin/bash -c "source /SMARTS/.venv/bin/activate"
RUN cd /SMARTS && pip install -e .


# XXX: This is an optimization since prior we were uploading the pip
# dependencies + our hiway package alongside the scoring program.
# Putting it here keeps the file size smaller. Alternatively, with buy-in we
# could install the hiway package directly if we are ok w/ distributing it
# this way. These requirements are generated via ./gen_requirements_txt.sh
# COPY docker/requirements.txt /tmp/requirements.txt
# RUN pip install --no-cache-dir -r /tmp/requirements.txt

# XXX: We deal w/ this issue on CodaLab: https://github.com/tensorflow/tensorflow/issues/19584
# RUN pip install https://github.com/mdsimmo/tensorflow-community-wheels/releases/download/1.13.1_cpu_py3_6_amd64/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl

# Add https://github.com/krallin/tini
ARG TINI_VERSION=v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

#------------------------------------------------------------------------------
# Entrypoint
#------------------------------------------------------------------------------
ENTRYPOINT ["/tini", "--", "entrypoint.sh"]
CMD ["/bin/bash"]
# Once in container can then run,
#   python3 examples/competition
