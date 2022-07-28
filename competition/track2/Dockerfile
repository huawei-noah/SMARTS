# [User configurable] 
# Set your desired base image
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# [Do not modify]
ARG DEBIAN_FRONTEND=noninteractive

# [Do not modify]
# Install libraries required for SMARTS.
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

# [Do not modify]
# Update default python version.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# [Do not modify]
# Setup virtual environment and install pip.
ENV VIRTUAL_ENV=/opt/.venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip wheel

# [Do not modify]
# Install requirements.txt .
COPY ./submission/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# [Do not modify]
# Copy all source files.
COPY . /track2

# [Do not modify]
# The training code should automatically run upon executing `docker run` command.
ENTRYPOINT python3.8 /track2/train/train.py