#!/bin/bash -eu
DIR="$1"
if [ ! -e "$DIR" ]
  then
     echo "Directory does not exist!"
     exit 4
elif [ ! -d "$DIR" ]
  then
     echo "Not a Directory"
     exit 5
else
     echo "Nice! mounting to the docker container..."
fi
readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}")")
# Collect proxy environment variables to pass to docker build
function build_args_proxy() {
  # Enable host-mode networking if necessary to access proxy on localhost
  env | grep -iP '^(https?|ftp)_proxy=.*$' | grep -qP 'localhost|127\.0\.0\.1' && {
    echo -n "--network host "
  }
  # Proxy environment variables as "--build-arg https_proxy=https://..."
  env | grep -iP '^(https?|ftp|no)_proxy=.*$' | while read env_proxy; do
    echo -n "-e ${env_proxy} "
  done
}
echo $(build_args_proxy)
XAUTH=/tmp/.docker.xauth
docker run \
       -it \
       -d $(build_args_proxy) \
       --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
       --privileged \
       --env="XAUTHORITY=$XAUTH" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume=/usr/lib/nvidia-384:/usr/lib/nvidia-384 \
       --volume=/usr/lib32/nvidia-384:/usr/lib32/nvidia-384 \
       --runtime=nvidia \
       --device /dev/dri \
       --volume=$DIR:/SMARTS \
       --name=ultra \
       ultra:gpu
