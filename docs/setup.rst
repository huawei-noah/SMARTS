.. _set_up_smarts:

Setting up SMARTS
=================

=============
Prerequisites
=============

+ python3 (>=3.7)
+ ubuntu (>=16.04)

============
Installation
============

Run the following commands to setup the SMARTS simulator.

.. code-block:: bash

    $ git clone https://github.com/huawei-noah/SMARTS.git
    $ cd <path/to/SMARTS>

    # For latest stable release
    $ git checkout master

.. note::

    Alternatively, to use the current development (i.e., bleeding edge) version:

    .. code-block:: bash

        $ git checkout develop

.. code-block:: bash

    # For Mac OS X users, ensure XQuartz is pre-installed.
    # Install the system requirements. You may use the `-y` option to enable
    # automatic assumption of "yes" to all prompts to avoid timeout from 
    # waiting for user input. 
    $ bash utils/setup/install_deps.sh

    # Setup virtual environment. Presently at least Python 3.7 and higher is
    # officially supported.
    $ python3.7 -m venv .venv

    # Enter virtual environment to install dependencies.
    $ source .venv/bin/activate

    # Upgrade pip.
    $ pip install --upgrade pip

    # Install smarts with extras as needed. Extras include the following: 
    # `camera_obs` - needed for rendering camera observations, and for testing.
    # `test` - needed for testing.
    # `train` - needed for RL training and testing.
    $ pip install -e '.[camera_obs,test,train]'

    # Run sanity-test and verify they are passing.
    # If tests fail, check './sanity_test_result.xml' for test report. 
    $ make sanity-test

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
