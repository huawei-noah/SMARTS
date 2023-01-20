.. _diagnostic:

Diagnostic
==========

A development tool designed to

+ Track the performance of each SMARTS version.
+ Test the effects of improvements or optimizations made in SMARTS.

Setup
-----

Dump different number of actors, with different types, on a map without visualization and show the mean & standard deviation of steps per second. Available scenarios:

+ n social agents: 1, 10, 20, 50
+ n data replay actors: 1, 10, 20, 50, 200
+ n sumo traffic actors: 1, 10, 20, 50, 200
+ 10 agents to n data replay actors: 1, 10, 20, 50
+ 10 agent to n roads: 1, 10, 20, 50

Run
---

Run the diagnostic :func:`~smarts.diagnostic.run.main` with one or multiple scenarios, from the ``SMARTS/smarts/diagnostic`` folder.

.. code-block:: bash

    $ cd <path/to>/SMARTS
    $ scl diagnostic run <scenarios/path> [scenarios/path]
    # e.g., scl diagnostic run n_sumo_actors/1_actors