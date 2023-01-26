.. _containers:

Containers
==========

Docker
------

SMARTS docker images are hosted at `dockerhub <https://hub.docker.com/u/huaweinoah>`_.

.. code-block:: bash

    $ cd </path/to/SMARTS>
    $ docker run --rm -it -v $PWD:/src -p 8081:8081 huaweinoah/smarts:<version>
    # E.g. docker run --rm -it -v $PWD:/src -p 8081:8081 huaweinoah/smarts:v0.5.1

    # If visualization is needed, run Envision server in the background.
    $ scl envision start -p 8081 &

    # Build the scenario. 
    # This step is required on the first time, and whenever the scenario is modified.
    $ scl scenario build scenarios/sumo/loop --clean

    # Run an example. 
    # Add --headless if visualisation is not needed.
    $ python examples/control/chase_via_points.py scenarios/sumo/loop

    # Visit http://localhost:8081 in the host machine to see the running simulation in Envision.

Singularity (Apptainer)
-----------------------

Instructions for running SMARTS within a `singularity <https://apptainer.org/>`_ container.

.. code-block:: bash

    $ cd </path/to/SMARTS>

    # Build container from definition file.
    $ sudo singularity build ./utils/singularity/smarts.sif ./utils/singularity/smarts.def

    # Use the container to build the required scenarios.
    $ singularity shell --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif
    # Inside the container
    Singularity> scl scenario build /src/scenarios/sumo/loop/
    Singularity> exit

    # Then, run the container using one of the following methods.

    # 1. Run container in interactive mode.
    $ singularity shell --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif
    # Inside the container
    Singularity> python3.7 /src/examples/control/chase_via_points.py /src/scenarios/sumo/loop/ --headless

    # 2. Run commands within the container from the host system.
    $ singularity exec --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif python3.7 /src/examples/control/chase_via_points.py /src/scenarios/sumo/loop/ --headless

    # 3. Run container instance in the background.
    $ singularity instance start --containall --bind ../SMARTS:/src ./utils/singularity/smarts.sif smarts_train /src/examples/control/chase_via_points.py /src/scenarios/sumo/loop/ --headless
