.. _cli: 

Command Line Interface
======================

A command line interface named ``scl`` is available to interact with SMARTS.

Examples of common usage are as follows.

.. code-block:: bash

   # Start envision server
   $ scl envision start

   # Build all scenarios under given directories
   $ scl scenario build-all ./scenarios

   # Rebuild a single scenario, replacing any existing generated assets
   $ scl scenario build --clean scenarios/sumo/loop

   # Clean generated scenario artifacts
   $ scl scenario clean scenarios/sumo/loop


.. click:: cli.cli:scl
   :prog: scl
   :nested: full