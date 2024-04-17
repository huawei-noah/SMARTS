.. _setup:

Setup
=====

Demo Video
----------

.. raw:: html

    <video controls="controls" width="640">
        <source src="https://raw.githubusercontent.com/smarts-project/smarts-project.github.io/master/assets/demo.mp4" type="video/mp4" />
    </video>

Prerequisites
-------------

+ ``python3 (3.8, 3.9, 3.10, 3.11)``
+ ``ubuntu (>=16.04)``

Installation
------------

Package
^^^^^^^

This includes SMARTS but none of the examples.

.. code-block:: bash

    # For Mac OS X users, ensure XQuartz is pre-installed.
    # Install the system requirements. You may use the `-y` option to enable
    # automatic assumption of "yes" to all prompts to avoid timeout from 
    # waiting for user input. 
    $ bash utils/setup/install_deps.sh

    # This should install the latest version of SMARTS from package index (generally PyPI).
    $ pip install 'smarts[camera-obs,sumo,examples]'


Development
^^^^^^^^^^^

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

    # Setup virtual environment. Presently at least Python 3.8 and higher is
    # officially supported.
    $ python3.8 -m venv .venv

    # Enter virtual environment to install dependencies.
    $ source .venv/bin/activate

    # Upgrade pip.
    $ pip install --upgrade pip

    # Install smarts with extras as needed. Extras include the following: 
    # `camera-obs` - needed for rendering camera observations, and for testing.
    # `sumo` - needed for using SUMO scenarios.
    # `test` - needed for running tests.
    # `example` - needed for running examples.
    # `--config-settings editable_mode=strict` - may be needed depending on version of setuptools. 
    #      See https://github.com/huawei-noah/SMARTS/issues/2090.
    $ pip install -e '.[camera-obs,sumo,test,examples]' --config-settings editable_mode=strict

    # Run sanity-test and verify they are passing.
    # If tests fail, check './sanity_test_result.xml' for test report. 
    $ make sanity-test
