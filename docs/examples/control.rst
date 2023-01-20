.. _control:

Control Theory
==============

Several agent control policies and agent :class:`~smarts.core.controllers.ActionSpaceType` are demonstrated. Run these examples as follows.

.. code-block:: bash

    $ cd <path>/SMARTS
    # Build the scenario `scenarios/sumo/loop`.
    $ scl scenario build scenarios/sumo/loop
    # Run SMARTS simulation with Envision display and `loop` scenario.
    $ scl run --envision examples/<script_name>.py scenarios/sumo/loop
    # Visit http://localhost:8081/ to view the experiment.


#. Chase Via Points

    + script: :examples:`control/chase_via_points.py`
    + Multi agent
    + ActionSpaceType: :attr:`~smarts.core.controllers.ActionSpaceType.LaneWithContinuousSpeed`

#. Trajectory Tracking

    + script: :examples:`control/trajectory_tracking.py`
    + ActionSpaceType: :attr:`~smarts.core.controllers.ActionSpaceType.Trajectory`

#. OpEn Adaptive Control

    + script: :examples:`control/ego_open_agent.py`
    + ActionSpaceType: :attr:`~smarts.core.controllers.ActionSpaceType.MPC`

#. Laner
   
    + script: :examples:`control/laner.py`
    + Multi agent
    + ActionSpaceType: :attr:`~smarts.core.controllers.ActionSpaceType.Lane`

#. Parallel Environments

    + script: :examples:`control/parallel_environment.py`
    + Multiple SMARTS environments in parallel
    + ActionSpaceType: :attr:`~smarts.core.controllers.ActionSpaceType.LaneWithContinuousSpeed`
