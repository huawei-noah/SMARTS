# Setup

ULTRA can be run natively on your machine, or through a Docker container. See instructions below for each method of setup.

## Setup Natively

The following steps can be ignored if you already have the SMARTS environment.
```sh
# 1 - Navigate to the SMARTS directory.
$ cd <path_to_SMARTS>/SMARTS

# 2 - Create a virtual environment.
$ python3.7 -m venv .ultra

# 3 - Activate virtual environment to install all dependencies.
$ source .ultra/bin/activate

# 4 - Install black for formatting (if you wish to contribute).
$ pip install black

# 5 - Install dependencies.
$ pip install -e .[train]
```
Whether you already had the SMARTS environment or not, install dill in your activated environment.
```sh
$ pip install dill
```

## Setup with Docker

Build the Docker images and run the container.
```sh
# 1 - Navigate to the ULTRA directory.
$ cd <path_to_SMARTS>/SMARTS/ultra

# 2 - Build the Docker images.
$ docker build -t <container_name> --network=host .

# 3 - Create the Docker container.
$ docker run \
        -it \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
        --privileged \
        --env="XAUTHORITY=/tmp/.docker.xauth" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume=/usr/lib/nvidia-384:/usr/lib/nvidia-384 \
        --volume=/usr/lib32/nvidia-384:/usr/lib32/nvidia-384 \
        --runtime=nvidia \
        --device /dev/dri \
        --volume=<path_to_SMARTS>:/SMARTS \
        --name=ultra \
        <container_name>
```
