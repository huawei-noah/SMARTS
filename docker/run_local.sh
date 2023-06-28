#!/bin/bash

# MSI
# source ~/workspace/SMARTS/docker/run_local.sh MSI platoon platoon

# Compute
# source ~/workspace/SMARTS/docker/run_local.sh COMPUTE platoon platoon

# Compute
# source ~/workspace/SMARTS/docker/run_local.sh COMPUTE drive drive


# Arguments
CURRENTDATETIME=`date +"%Y_%m_%d_%H_%M"`
SERVER=$1
NAME=$2
CONTAINER="${NAME}_${CURRENTDATETIME}"
IMAGE="${NAME}:${CURRENTDATETIME}"
COMMAND=$3 

echo "ContainerName and ImageName"
echo ${CONTAINER} 
echo ${IMAGE}

GPU=3,4
CPUs=0-7

# Collect proxy environment variables to pass to docker build
function build_args_proxy() {
  # Proxy environment variables as "--build-arg https_proxy=https://..."
  env | grep -iP '^(https?|ftp|no)_proxy=.*$' | while read env_proxy; do
    echo -n "--build-arg ${env_proxy} "
  done
}

readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}")")

case $SERVER in
    "MSI")
        SRC=$SCRIPT_DIR/../../testing
        GPUFLAG="--gpus '\"device=$GPU\"'"
        CPUFLAG="--cpuset-cpus='"$CPUs"'" 
        BUILDCONTEXT=$SCRIPT_DIR/../
        PROXY=""
        ;;   
    "COMPUTE")
        SRC=$SCRIPT_DIR/../../testing
        GPUFLAG="--gpus '\"device=$GPU\"'"
        CPUFLAG=""
        BUILDCONTEXT=$SCRIPT_DIR/../
        PROXY=$(build_args_proxy)
        ;;         
    *)
        echo "Unknown server"
        return 5
        ;;
esac

case $COMMAND in
    "platoon")
        COMMAND_PROGRAM="PYTHONHASHSEED=0 python3.8 ./examples/rl/platoon/train/run.py"
        DST="/SMARTS/examples/rl/platoon/train/logs"
        DOCKERFILE="${BUILDCONTEXT}/examples/rl/platoon/train/Dockerfile"
        ;;
    "drive")
        COMMAND_PROGRAM="PYTHONHASHSEED=0 python3.8 ./examples/rl/drive/train/run.py"
        DST="/SMARTS/examples/rl/drive/train/logs"
        DOCKERFILE="${BUILDCONTEXT}/examples/rl/drive/train/Dockerfile"
        ;;
    *)
        echo "Unknown program command"
        return 5
        ;;
esac

# Build Docker image locally
DOCKER_BUILDKIT=1 docker build \
    -t ${IMAGE} \
    --network=host \
    ${PROXY} \
    -f ${DOCKERFILE} \
    ${BUILDCONTEXT}

# Run docker container in detached tmux terminal
COMMAND_DOCKER="docker run --rm -it \
    --network=host \
    -e http_proxy \
    -e https_proxy \
    -e HOST_NAME=$(id -un) \
    -e HOST_UID=$(id -u) \
    -e HOST_GID=$(id -g) \
    ${CPUFLAG} \
    ${GPUFLAG} \
    --volume=${SRC}:${DST} \
    ${IMAGE} bash
"

tmux new -d -s ${CONTAINER}
tmux send-keys -t ${CONTAINER} "${COMMAND_DOCKER}" ENTER
tmux send-keys -t ${CONTAINER} "${COMMAND_PROGRAM}" ENTER
