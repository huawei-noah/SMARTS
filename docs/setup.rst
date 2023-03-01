.. _setup:

Setup
=====

Prerequisites
-------------

+ python3 (3.7 or 3.8)
+ ubuntu (>=16.04)

Installation
------------

Run the following commands to setup the SMARTS simulator.

.. code-block:: bash

    $ git clone https://github.com/huawei-noah/SMARTS.git
    $ cd <path/to/SMARTS>

    # For latest stable release
    $ git checkout tags/<tag_name>
    # e.g., git checkout tags/v1.0.3

.. note::

    Alternatively, to use the current development (i.e., bleeding edge) version:

    .. code-block:: bash

        $ git checkout master

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