.. _driving_smarts_2022:

Driving SMARTS 2022
===================

The Driving SMARTS 2022 is a benchmark derived from the
`NeurIPS 2022 Driving SMARTS <https://smarts-project.github.io/archive/2022_nips_driving_smarts/>`_ competition.

This benchmark is intended to address the following requirements:

-  The benchmark should use an up-to-date version of ``gym`` to simplify the
   interface. As such we should use `gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ for new environments.
-  The competition gym environment should strictly follow the
   requirements of the gym interface to allow for obvious actions and
   observations.
-  The observations out of this environment should be just data.
-  The benchmark runner should be configurable for future benchmark
   versions.
-  Added benchmarks should be versioned.
-  Added benchmarks should be discover-able.

This benchmark allows ego agents to use any one of the following action spaces.

+ :attr:`~smarts.core.controllers.ActionSpaceType.TargetPose`
+ :attr:`~smarts.core.controllers.ActionSpaceType.RelativeTargetPose`

See the list of :ref:`available zoo agents <available_zoo_agents>` which are compatible with this benchmark. A compatible zoo agent can be run as follows.

.. code-block:: bash

    $ cd <path>/SMARTS
    $ scl zoo install <agent path>
    # e.g., scl zoo install zoo/policies/interaction_aware_motion_prediction
    $ scl benchmark run driving_smarts==0.0 <agent_locator> --auto_install
    # e.g., scl benchmark run driving_smarts==0.0 zoo.policies:interaction-aware-motion-prediction-agent-v0 --auto-install