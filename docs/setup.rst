.. _set_up_smarts:

Setting up SMARTS
=================

==========
First Steps
==========

To setup the simulator, which is called SMARTS, run the following commands,

.. code-block:: bash

    # git clone ...
    cd <project>

    # Follow the instructions given by prompt for setting up the SUMO_HOME environment variable
    ./install_deps.sh

    # verify sumo is >= 1.5.0
    # if you have issues see ./doc/SUMO_TROUBLESHOOTING.md
    sumo

    # setup virtual environment; presently only Python 3.7.x is officially supported
    python3.7 -m venv .venv

    # enter virtual environment to install all dependencies
    source .venv/bin/activate

    # upgrade pip, a recent version of pip is needed for the version of tensorflow we depend on
    pip install --upgrade pip

    # install [train] version of python package with the rllib dependencies
    pip install -e .[train]

    # make sure you can run tests (and verify they are passing)
    make test

    # then you can run a scenario, see following section for more details

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

Check out the paper at `Link SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving <https://arxiv.org/abs/2010.09776>`_ for background on some of the project goals.

========
CLI Tool
========

SMARTS provides a command-line tool to interact with scenario studio and Envision.

Usage

.. code-block:: bash

    scl COMMAND SUBCOMMAND [OPTIONS] [ARGS]...

Commands:

- envision
- scenario
- zoo

Subcommands of scenario:

- build-all: Generate all scenarios under the given directories
- build: Generate a single scenario
- clean: Clean generated artifacts

Subcommands of envision:

- start: Start envision server

Subcommands of zoo:

- build: Build a policy