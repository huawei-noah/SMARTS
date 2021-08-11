# Setup

ULTRA can be run natively on a linux machine, or through a Docker container. ULTRA also supports installation on Windows Subsystem for Linux (i.e. WSL 1 and WSL 2). See instructions below for each method of setup.

## Setup Natively on a Linux machine and/or WSL

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

# 2 - Build the Docker image.
$ docker build --no-cache --network=host -f Dockerfile -t ultra:latest .

# 3 - Create and run the Docker container.
$ docker run --rm -it -v $(PWD):/src -p 8081:8081 ultra:latest
```
