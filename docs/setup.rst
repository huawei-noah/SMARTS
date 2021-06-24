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

    # OPTIONAL: install [camera-obs] version of python package with the panda3D dependencies if you want to render camera sensor observations in your simulations
    pip install -e .[camera-obs]

    # make sure you can run sanity-test (and verify they are passing)
    # if tests fail, check './sanity_test_result.xml' for test report.
    pip install -e .[test]
    make sanity-test

    # then you can run a scenario, see following section for more details

================
Running
================

We use the `scl` command line to run SMARTS together with it's supporting processes. To run the default example simply build a scenario and run the following command:

.. code-block:: bash
    # build scenarios/loop
    scl scenario build --clean scenarios/loop

    # run an experiment
    scl run --envision examples/single_agent.py scenarios/loop


You need to add the `--envision` flag to run the Envision server where you can see the visualization of the experiment. See [./envision/README.md](./envision/README.md) for more information on Envision, our front-end visualization tool.

After executing the above command, visit http://localhost:8081/ in your browser to view your experiment.


Several example scripts are provided under [`SMARTS/examples`](./examples), as well as a handful of scenarios under [`SMARTS/scenarios`](./scenarios). You can create your own scenarios using the [Scenario Studio](./smarts/sstudio). Below is the generic command to run and visualize one of the example scripts with a scenario.

.. code-block:: bash
    scl run --envision <examples/script_path> <scenarios/path>


Pass in the agent example path and scenarios folder path above to run an experiment like the one mentioned above.

================
Examples
================

.. code-block:: bash
    # Start envision, serve scenario assets out of ./scenarios
    scl envision start --scenarios ./scenarios

    # Build all scenario under given directories
    scl scenario build-all ./scenarios ./eval_scenarios

    # Rebuild a single scenario, replacing any existing generated assets
    scl scenario build --clean scenarios/loop

    # Clean generated scenario artifacts
    scl scenario clean scenarios/loop


================
Troubleshooting
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

2. SUMO
SUMO can have some problems in setup. Please look through the following for support for SUMO:

If you are having issues see: **[SETUP](docs/setup.rst)** and **[SUMO TROUBLESHOOTING](docs/SUMO_TROUBLESHOOTING.md)**.

* If you wish to find binaries: **[SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php )**
* If you wish to compile from source see: **[SUMO Build Instructions](https://sumo.dlr.de/docs/Developer/Main.html#build_instructions )**.
* **Please note that building SUMO may not install other vital dependencies that SUMO requires to run.**
* If you build from the git repository we recommend to use **[SUMO version 1.7.0](https://github.com/eclipse/sumo/commits/v1_7_0)** or newer.

3. General
In many cases additional run logs are located at `~/.smarts`. These can sometimes be helpful.

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

- envision
- scenario
- zoo
- run

Subcommands of scenario:

- build-all: Generate all scenarios under the given directories
- build: Generate a single scenario
- clean: Clean generated artifacts

Subcommands of envision:

- start: Start envision server

Subcommands of zoo:

- build: Build a policy

Subcommands of run:
No subcommands of `run`. You can directly use `run` to simulate an experiment as mentioned in the example above.