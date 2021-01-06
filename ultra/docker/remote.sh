#!/bin/bash

# An example to execute this script:
# source remote.sh <user name> <password> <server name> <container name>
# source remote.sh z84216771 abcd1234 GX3 ultratest

# Servers
# CX3 = 10.193.241.233
# CX4 = 10.193.241.234
# Compute-4 = 10.193.192.17
# Compute-11 = 10.193.192.113
# GX3 = 10.193.241.239

# Arguments
USER=$1       #Username, e.g., z84216771
PASS=$2       #Password, e.g., abcd1234
SERVER=$3     #Server, e.g., GX3
CONTAINER=$4  #Container name, e.g., ultratest

case $SERVER in
    "CX3")
        ADD=10.193.241.233
        DST=/data/research
        GPUFLAG=""
        ;;
    "CX4")
        ADD=10.193.241.234
        DST=/data/research
        GPUFLAG=""
        ;;
    "Compute-4")
        ADD=10.193.192.17
        DST=/data/$USER
        GPUFLAG="--runtime=nvidia"        
        ;;
    "Compute-11")
        ADD=10.193.192.113
        DST=/data/$USER
        GPUFLAG="--runtime=nvidia"
        ;;
    "GX3")
        ADD=10.193.241.239
        DST=/data/research
        GPUFLAG="--gpus=all"
        ;;
    *)
        echo "Unknown server"
        return 5
        ;;
esac

# Build Docker image locally
docker build \
    -t ${CONTAINER} \
    --network=host \
    .
# Save image locally
docker save -o ./${CONTAINER}.tar ${CONTAINER}
# Push image to remote server
sshpass -p ${PASS} scp ${PWD}/${CONTAINER}.tar ${USER}@${ADD}:${DST}
# Load docker in remote server
sshpass -p ${PASS} ssh ${USER}@${ADD} "docker load -i ${DST}/${CONTAINER}.tar"
# Run interactive docker container in detached mode in remote server 
sshpass -p ${PASS} ssh ${USER}@${ADD} "docker run --rm \
    -it \
    -d \
    --network=host \
    --privileged \
    --env="XAUTHORITY=/tmp/.docker.xauth" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume=/usr/lib/nvidia-384:/usr/lib/nvidia-384 \
    --volume=/usr/lib32/nvidia-384:/usr/lib32/nvidia-384 \
    --volume=${DST}/logs:/ULTRA/logs \
    --volume=/etc/localtime:/etc/localtime:ro \
    ${GPUFLAG} \
    --device=/dev/dri \
    --memory=100g \
    --user=$(id -u):$(id -g) \
    --name=${CONTAINER} \
    ${CONTAINER}
"

# Enter the interactive docker container
# docker exec -ti ${CONTAINER} bash
# Exit the interactive docker container
# ctrl-p ctrl-q