#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
endloc=${SCRIPT_DIR}/build

echo 'If you are building for your system please check with "cat /etc/os-release | grep VERSION_"'

# Parse command line arguments
options=$(getopt -o u:s: --long ubuntu-version:,sumo-tag: -- "$@")
eval set -- "$options"

# Default values
ubuntu_version=""
sumo_tag=""

# Loop through arguments and set variables
while true; do
  case "$1" in
        -u | --ubuntu-version )
          ubuntu_version="$2"
          shift 2
          ;;
        -s | --sumo-tag )
          sumo_tag="$2"
          shift 2
          ;;
        -- )
          shift
          break
          ;;
        * )
          echo "Invalid argument: $1"
          exit 1
          ;;
  esac
done

# Check for empty arguments
if [ -z "$ubuntu_version" ] || [ -z "$sumo_tag" ]; then
  echo "Error: both ubuntu-version and sumo-tag options are required"
  echo "-u, --ubuntu-version <ubuntu docker tag e.g. xenial, 16.04, 21.04, ...>"
  echo "       See https://hub.docker.com/_/ubuntu/tags"
  echo "-s, --sumo-tag <sumo commit tag or commit id>"
  echo "       See https://github.com/eclipse/sumo/tags"
  exit 1
fi


# Print arguments
echo "Ubuntu version: $ubuntu_version"
echo "SUMO tag: $sumo_tag"


docker build --no-cache --build-arg SUMO_REPO_TAG=${sumo_tag} --build-arg UBUNTU_VERSION=${ubuntu_version} -t sumo_builder - < ${SCRIPT_DIR}/Dockerfile.sumo_build && \
docker create -ti --name intermediate_builder sumo_builder /bin/bash
if docker cp intermediate_builder:/usr/src/sumo ${endloc} && \
docker rm -f intermediate_builder ; then
    echo 'Current SUMO_HOME is:'
    printenv | grep 'SUMO_HOME' 
    read -p "Do you wish to set SUMO_HOME to ${endloc}? [Yn]" should_change_sumo_home
    if [[ $should_change_sumo_home =~ ^[yY]$ ]]; then
        # should not be doing this if not linux
        if ! [[ "$OSTYPE" == "linux-gnu" ]]; then
            echo 'Not Linux, exiting...'
            exit 1
        fi
        # delete current SUMO_HOME from bashrc
        sed '/^export SUMO_HOME/d' ~/.bashrc
        echo "export SUMO_HOME=${endloc}" >> ~/.bashrc
        echo "We've updated your ~/.bashrc. Be sure to run:"
        echo ""
        echo "  source ~/.bashrc"
        echo ""
        echo "in order to set the SUMO_HOME variable in your current console session"
    fi
fi
