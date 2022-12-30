.. _set_up_smarts:

Setting up SMARTS
=================

============
Installation
============

Run the following commands to setup the SMARTS simulator.

.. code-block:: bash

    git clone https://github.com/huawei-noah/SMARTS.git
    cd <path/to/SMARTS>

    # For Mac OS X users, ensure XQuartz is pre-installed.
    # Install the system requirements. You may use the `-y` option to enable automatic assumption of "yes" to all prompts to avoid timeout from waiting for user input. 
    bash utils/setup/install_deps.sh

    # Setup virtual environment. Presently at least Python 3.7 and higher is officially supported.
    python3.7 -m venv .venv

    # Enter virtual environment to install dependencies.
    source .venv/bin/activate

    # Upgrade pip.
    pip install --upgrade pip

    # Install smarts with extras as needed. Extras include the following: 
    # `camera_obs` - needed for rendering camera sensor observations, and for testing.
    # `test` - needed for testing.
    # `train` - needed for RL training and testing.
    pip install -e '.[camera_obs,test,train]'

    # Run sanity-test and verify they are passing.
    # If tests fail, check './sanity_test_result.xml' for test report. 
    make sanity-test

=======
Running
=======

Use the `scl` command to run SMARTS together with it's supporting processes. 

To run the default example, firstly build the scenario `scenarios/sumo/loop`.

.. code-block:: bash

    $ scl scenario build --clean scenarios/sumo/loop

Then, run a single-agent SMARTS simulation with Envision display and `loop` scenario.

.. code-block:: bash
    
    $ scl run --envision examples/single_agent.py scenarios/sumo/loop 

The `--envision` flag runs the Envision server which displays the simulation visualization. See Envision's README(./envision/README.md) for more information on Envision, SMARTS's front-end visualization tool.

After executing the above command, visit `http://localhost:8081/ <http://localhost:8081/>`_ to view the experiment.

Several example scripts are provided in [examples](./examples) folder, as well as a handful of scenarios in [scenarios](./scenarios) folder. You can create your own scenarios using the [Scenario Studio](./smarts/sstudio). Below is the generic command to run and visualize one of the example scripts with a scenario.

.. code-block:: bash
    
    scl run --envision <examples/path> <scenarios/path>


================
Examples
================

.. code-block:: bash
    
    # Start envision, serve scenario assets out of ./scenarios
    scl envision start --scenarios ./scenarios

    # Build all scenario under given directories
    scl scenario build-all ./scenarios ./eval_scenarios

    # Rebuild a single scenario, replacing any existing generated assets
    scl scenario build --clean scenarios/sumo/loop

    # Clean generated scenario artifacts
    scl scenario clean scenarios/sumo/loop


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
SUMO might encounter problems during setup. Please look through the following for support for SUMO:

If you are having issues see: **[SETUP](docs/setup.rst)** and **[SUMO TROUBLESHOOTING](docs/SUMO_TROUBLESHOOTING.md)**.

* If you wish to find binaries: **[SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php )**
* If you wish to compile from source see: **[SUMO Build Instructions](https://sumo.dlr.de/docs/Developer/Main.html#build_instructions )**.
* **Please note that building SUMO may not install other vital dependencies that SUMO requires to run.**
* If you build from the git repository we recommend to use **[SUMO version 1.7.0](https://github.com/eclipse/sumo/commits/v1_7_0)** or newer.

3. General
In many cases additional run logs are located at `~/.smarts`. These can sometimes be helpful.
