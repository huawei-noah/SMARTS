endloc=${PWD}/ubuntu_xenial_build

docker build -t smrt_u16_sumo_1_5 - < Dockerfile.ubuntu16.sumo1_5 && \
docker create -ti --name intermediate_builder smrt_u16_sumo_1_5 /bin/bash
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
        # should not be doing this if not Xenial
        if ! cat /etc/os-release | grep 'Xenial' ; then
            echo 'Not Ubuntu 16(Xenial), exiting...'
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