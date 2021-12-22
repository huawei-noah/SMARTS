#!/usr/bin/env bash

function check_python_version_gte_3_7 {
    echo "Checking for >=python3.7"
    # running through current minor verions
    hash python3.7 2>/dev/null \
    || hash python3.8 2>/dev/null \
    || hash python3.9 2>/dev/null;
}

function do_install_for_linux {
    echo "Installing sumo (used for traffic simulation and road network)"
    sudo add-apt-repository ppa:sumo/stable
    sudo apt-get update

    sudo apt-get install -y \
         libspatialindex-dev \
         sumo sumo-tools sumo-doc \
         build-essential cmake

    #only a problem for linux
    if ! check_python_version_gte_3_7; then

         echo "A >=3.7 python version not found"
         read -p "Install python3.7? [Yn]" should_add_python_3_7
         if [[ $should_add_python_3_7 =~ ^[yY\w]*$ ]]; then
              echo ""
              printf "This will run the following commands:\n$ sudo apt-get update\n$ sudo apt-get install software-properties-common\n$ sudo add-apt-repository ppa:deadsnakes/ppa\n$ sudo apt-get install python3.7 python3.7-dev python3.7-tk python3.7-venv"
              echo ""
              read -p "WARNING. Is this OK? If you are unsure choose no. [Yn]" should_add_python_3_7
              # second check to make sure they really want to
              if [[ $should_add_python_3_7 =~ ^[yY\w]*$ ]]; then
                    sudo apt-get install software-properties-common
                    sudo add-apt-repository ppa:deadsnakes/ppa
                    sudo apt-get install python3.7 python3.7-dev python3.7-tk python3.7-venv
              fi
         fi
    fi

    echo ""
    echo "-- dependencies have been installed --"
    echo ""
    echo "You'll need to set the SUMO_HOME variable. Logging out and back in will"
    echo "get you set up. Alternatively, in your current session, you can run:"
    echo ""
    echo "  source /etc/profile.d/sumo.sh"
    echo ""
}

function do_install_for_macos {
    echo "Installing sumo (used for traffic simulation and road network)"
    brew tap dlr-ts/sumo
    brew install sumo spatialindex # for sumo
    brew install geos # for shapely

    # start X11 manually the first time, logging in/out will also do the trick
    open -g -a XQuartz.app

    echo ""
    echo "-- dependencies have been installed --"
    echo ""
    read -p "Add SUMO_HOME to ~/.bash_profile? [Yn]" should_add_SUMO_HOME
    echo "should_add_SUMO_HOME $should_add_SUMO_HOME"
    if [[ $should_add_SUMO_HOME =~ ^[yY\w]*$ ]]; then
        echo 'export SUMO_HOME="/usr/local/opt/sumo/share/sumo"' >> ~/.bash_profile
        echo "We've updated your ~/.bash_profile. Be sure to run:"
        echo ""
        echo "  source ~/.bash_profile"
        echo ""
        echo "in order to set the SUMO_HOME variable in your current session"
    else
        echo "Not updating ~/.bash_profile"
        echo "Make sure SUMO_HOME is set before proceeding"
    fi
}

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    echo "Detected Linux"
    do_install_for_linux
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    do_install_for_macos
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi
