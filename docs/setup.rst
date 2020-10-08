.. _set_up_smarts:

Setting up SMARTS
=================

==========
First Steps
==========

To setup the simulator, which is called SMARTS, run the following commands,

.. code-block:: bash

    # unzip the starter_kit and place somewhere convenient on your machine. (e.x. ~/src/starter_kit)

    cd ~/src/starter_kit
    ./install_deps.sh
    # ...and, follow any on-screen instructions

    # test that the sumo installation worked
    sumo-gui

    # setup virtual environment (Python 3.7 is required)
    # or you can use conda environment if you like.
    python3.7 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip

    # install the dependencies
    pip install smarts-<version>-py3-none-any.whl
    pip install smarts-<version>-py3-none-any.whl[train]

    # download the public datasets from Codalab to ./dataset_public

    # test that the sim works
    python random_example/run.py

================
Common Questions
================

1. Exception: Could not open window.

This may be due to some old dependencies of Panda3D. Try the following instructions to solve it.

.. code-block:: bash

    # set DISPLAY 
    vim ~/.bashrc
    export DISPLAY=":1"
    source ~/.bashrc

    # set xorg server
    sudo wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf
    sudo /usr/bin/Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ./xdummy.log -config /etc/X11/xorg.conf $DISPLAY & 0

====
Docs
====

To look at the documentation call:

.. code-block:: bash

    # Browser will attempt to open on localhost:8082
    scl docs

========
CLI Tool
========

SMARTS provides a command-line tool to interact with scenario studio and Envision.

Usage

.. code-block:: bash

    scl COMMAND SUBCOMMAND [OPTIONS] [ARGS]...

Commands:
* envision
* scenario
* docs

Subcommands of scenario:
* build-all: Generate all scenarios under the given directories
* build: Generate a single scenario
* clean: Clean generated artifacts

Subcommands of envision:
* start: start envision server
