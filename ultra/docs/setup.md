# Setup

ULTRA can be run natively on your machine, or through a Docker container. See instructions below for each method of setup.

## Setup Natively

```sh
# 1 - Navigate to the ULTRA directory.
$ cd <path_to_SMARTS>/SMARTS/ultra

# 2 - Install overhead dependencies (python3.7 & sumo)
$ ./install_deps.sh

# 3 - verify sumo is >= 1.5.0
# if you have issues see ./docs/SUMO_TROUBLESHOOTING.md
$ sumo

# 4 - Create a virtual environment.
$ python3.7 -m venv .venv

# 5 - Activate virtual environment to install all dependencies.
$ source .venv/bin/activate

# 6 - Install dependencies.
$ pip install -e .
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
