# ULTRA
Unprotected Left Turn using Reinforcement-learning Agents
---
Ultra provides a gym-based environment using SMARTS for tackling intersection navigation and more specifically unprotected left turn.

Here is the summary of key features:
 - creating customized scenarios with different levels of difficulty
 - defining/analyzing traffic designs including low-mid-high densities
 - defining/analyzing social-vehicle behaviors
 - configurable train/test parameters
 - train/test custom RL algorithms against benchmark results

### Benchmark
Read the [documentation](https://gitlab.smartsai.xyz/smarts/ULTRA/-/wikis/Benchmark)

### Setup
Ignore this step if you already have the SMARTS environment installed.
#### Setup Natively
  ```sh
  python3.7 -m venv .ultra
  # 1-activate virtual environment to install all dependencies
  source .smarts/bin/activate
  # 2-install black for formatting (if you wish to contribute)
  pip install black
  # 3-install dependencies
  pip install -e .[train]
  ```

#### Setup with Docker
- SMARTS is pre-installed and ULTRA source files are copied automatically into Docker image.
- Build a docker image alone
  ```sh
  $ cd path/to/repository/SMARTS/ultra
  $ docker build -t <container name> --network=host . # or run make
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
         --volume=$DIR:/SMARTS \ #  fill $DIR with the path to SMARTS to mount
         --name=ultra \
         ultra:gpu
  ```


### Server
- Server name, address, and destination path `${DST}` for storage

  |Server|Address|${DST}|
  |:----|:----|:----|
  |CX3|10.193.241.233|/data/research|
  |CX4|10.193.241.234|/data/research|
  |Compute-4|10.193.192.17|/data/$USER|
  |Compute-11|10.193.192.113|/data/$USER|
  |GX3|10.193.241.239|/data/research|  

### Get Started with ULTRA
- [Start training a baseline agent](docs/getting_started.md)
- [Create a custom agent](docs/custom_agent.md)
