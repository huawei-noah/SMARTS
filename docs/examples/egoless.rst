.. _egoless:

Egoless
=======

Simulate a SMARTS environment without any ego agents, but with only background traffic. Run these examples as follows.

.. code-block:: bash

    $ cd <path>/SMARTS
    # Build the scenario `scenarios/sumo/loop`.
    $ scl scenario build scenarios/sumo/loop
    # Run SMARTS simulation with Envision display and `loop` scenario.
    $ scl run --envision examples/<script_name>.py scenarios/sumo/loop
    # Visit http://localhost:8081/ to view the experiment.


#. Egoless

    + script: :examples:`egoless.py`
