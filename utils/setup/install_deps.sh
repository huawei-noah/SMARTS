#!/usr/bin/env bash

function check_python_version_gte_3_8 {
    echo "Checking for >=python3.8"
    # running through current minor versions
    hash python3.8 2>/dev/null \
    || hash python3.8 2>/dev/null \
    || hash python3.9 2>/dev/null;
}

function install_python_3_8 {
    echo "Installing python3.8"
    sudo apt-get install $1 software-properties-common
    sudo add-apt-repository $1 ppa:deadsnakes/ppa
    sudo apt-get install $1 python3.8 python3.8-tk python3.8-venv
}

function do_install_for_linux {
    echo "Installing dependencies"
    sudo apt-get update
    sudo apt-get install $1 \
        libspatialindex-dev \
        build-essential cmake

    #only a problem for linux
    if ! check_python_version_gte_3_8; then
        echo "A >=3.8 python version not found"
        if  [[ "$1" = "-y" ]]; then
            install_python_3_8 "$1"
        else
            read -p "Install python3.8? [Yn]" should_add_python_3_8
            if [[ $should_add_python_3_8 =~ ^[yY\w]*$ ]]; then
                echo ""
                printf "This will run the following commands:\n$ sudo apt-get update\n$ sudo apt-get install software-properties-common\n$ sudo add-apt-repository ppa:deadsnakes/ppa\n$ sudo apt-get install python3.8 python3.8-tk python3.8-venv"
                echo ""
                read -p "WARNING. Is this OK? If you are unsure choose no. [Yn]" should_add_python_3_8
                # second check to make sure they really want to
                if [[ $should_add_python_3_8 =~ ^[yY\w]*$ ]]; then
                    install_python_3_8
                fi
            fi
        fi
    fi

    echo ""
    echo "-- dependencies have been installed --"
    echo ""
}

function do_install_for_macos {
    echo "Installing denpendencies"
    brew install spatialindex
    brew install geos # for shapely

    # start X11 manually the first time, logging in/out will also do the trick
    open -g -a XQuartz.app

    echo ""
    echo "-- dependencies have been installed --"
    echo ""
}

if  [[ "$1" = "-y" ]]; then
    echo "Automatic \"yes\" assumed for all prompts and runs non-interactively."
fi
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    echo "Detected Linux"
    do_install_for_linux "$1"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    do_install_for_macos "$1"
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi
