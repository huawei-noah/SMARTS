SMARTS Command Line 
====================

SMARTS provides a command-line tool to interact with scenario studio and Envision.

Usage

.. code-block:: bash

    scl COMMAND SUBCOMMAND [OPTIONS] [ARGS]...

Commands:

- envision
- scenario
- zoo
- run

Each command and subcommand has a --help for more information about the subcommand and options


scenario:
==========

Subcommands:

- build-all <path-to-scenarios>: Generate all scenarios under the given directories
- build <path-to-scenario>: Generate a single scenario
- clean <path-to-scenario(s)>: Clean generated artifacts 

Options: (build and build-all)
- clean: Clean previously generated artifacts first
- allow-offset-map(s): Allopw road networks (maps) to be offset from the origin. If not specified,
creates a new network file if necessary

ex. Build and clean a single scenario

.. code-block:: bash

    scl scenario build --clean path/to/scenario/directory

Scenarios can reference remote packages by including a requirements.txt file in the root of the scenario directory.
Packages will be installed during the build.

In the requirements.txt file:

.. code-block:: bash

    --extra-index-url http://localhost:8080
    <dependency>==1.0.0
    ...


envision:
==========

Subcommands:

- start: Start envision server

Options:

- -p, --port INTEGER: Specify envision port to use (default=8081)
- -s, --scenarios TEXT: A list of directories where scenarios are stored
- -c, --max_capacity FLOAT: Max capacity in MB of Envision's playback buffer.

ex. Start envision with custom port 1000

.. code-block:: bash

    scl envision start --port 1000


zoo:
=====

Subcommands:

- build <path-to-policy>: Build a policy
- install TEXT: Attempt to install the specified agents from the given paths/url

ex. Build the rl-agent policy

.. code-block:: bash

    scl zoo build SMARTS/zoo/policies/rl-agent

Local zoo agent packages can be built into wheels using a setup.py and requirements.txt file.
To use policies in scenarios, create a requirements.txt in the scenario root

.. code-block:: bash
    --extra-index-url http://localhost:8080
    rl-agent==1.0.0


run:
=====

Subcommands:
No subcommands of `run`. You can directly use `run` to simulate an experiment as mentioned in the example above.

Options:

- --envision: start up with an Envision server
- -p, --envision_port TEXT: Port on which Envision will run

ex. Run an experiment with Envision enabled

.. code-block:: bash

    scl run examples/single_agent.py scenarios/loop --envision