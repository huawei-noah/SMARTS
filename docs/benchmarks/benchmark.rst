.. _benchmark:

Driving SMARTS 2022
===================

The Driving SMARTS 2022 is a benchmark derived from the
NeurIPS 2022 Driving SMARTS Competition.

This benchmark is intended to address the following requirements:

-  The benchmark should use an up to date version of gym to simplify the
   interface. (we used gymnasium)
-  The competition gym environment should strictly follow the
   requirements of the gym interface to allow for obvious actions and
   observations.
-  The observations out of this environment should be just data.
-  The benchmark runner should be configurable for future benchmark
   versions.
-  Added benchmarks should be versioned.
-  Added benchmarks should be discover-able.

See: https://smarts-project.github.io/archive/2022_nips_driving_smarts/
for historical context.

Benchmark discovery
-------------------

The ``scl benchmark list`` command is used to check what benchmarks are
available.

.. code:: bash

   $ scl benchmark list 
   BENCHMARK_NAME               BENCHMARK_ID             VERSIONS
   - Driving SMARTS:            driving_smarts           0.0 0.1

If creating your own version it is also possible to use
``--benchmark-listing`` to target a different benchmark listing file.

Agent configuration
-------------------

Agent Configuration for Driving SMARTS 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current version of the agent configuration is as follows:

.. code:: yaml

   # baselines/driving_smarts/v0/agent_config.yaml
   agent:
     interface: # This is a section specific to the benchmark `driving_smarts==0.0`
       # we currently allow `RelativeTargetPose` and `TargetPose` 
       action_space: "RelativeTargetPose" 
       img_meters: 50 # Observation image area size in meters.
       img_pixels: 112 # Observation image size in pixels.
     locator: "smarts.zoo:random-relative-target-pose-agent-v0" # This is an example agent
     kwargs: # A collection of keyword arguments provided to the agent entrypoint
       speed: 20

Of particular note is the ``locator`` which is has to resolve to the
registration of an agent. This can be registered as seen below.

.. code:: python

   # smarts/zoo/__init__.py
   import math
   import random
   from typing import Any, Dict

   from smarts.core.agent import Agent
   from smarts.core.agent_interface import ActionSpaceType, AgentInterface
   from smarts.zoo.agent_spec import AgentSpec
   from smarts.zoo.registry import register


   class RandomRelativeTargetPoseAgent(Agent):
       """A simple agent that can move a random distance."""

       def __init__(self, speed=28, timestep=0.1) -> None:
           super().__init__()
           self._speed_per_step = speed / timestep

       def act(self, obs: Dict[str, Any], **configs):
           return [
               (random.random() - 0.5) * self._speed_per_step,
               (random.random() - 0.5) * self._speed_per_step,
               random.random() * 2 * math.pi - math.pi,
           ]

   # Note `speed` from configuration file maps here.
   def entry_point(speed=10, **kwargs):
       """An example entrypoint for a simple agent.
       This can have any number of arguments similar to the gym environment standard.
       """
       return AgentSpec(
           AgentInterface(
               action=ActionSpaceType.RelativeTargetPose,
           ),
           agent_builder=RandomRelativeTargetPoseAgent,
           agent_params=dict(speed=speed),
       )


   # Where the name of the agent is registered.
   # note this is in `smarts/zoo/__init__.py` which is the `smarts.zoo` module.
   # this would be referenced like `"smarts.zoo:random-relative-target-pose-agent-v0"`
   register("random-relative-target-pose-agent-v0", entry_point)

The syntax of the referencing the locator is like ``"``
``module.importable.in.python`` ``:`` ``registered_name_of_agent``
``-v`` ``X`` ``"``.

-  Module: ``module.importable.in.python`` The module section must be
   importable from within python. An easy test to see if the module is
   importable is to try importing the module within interactive python
   or a script (e.g. ``import module.importable.in.python``)
-  Separator: ``:`` This separates the module and name sections of the
   locator.
-  Registered name: ``registered_name_of_agent`` The name of the agent
   as registered using ``smarts.zoo.register``.
-  Version separator: ``-v`` This separates the name and version
   sections of the locator.
-  Version: ``X`` The version of the agent (this is required to register
   an agent.) ``X`` can be any integer.

Running the benchmark
---------------------

The easiest way to run the benchmark is through ``scl benchmark run``.
This takes a benchmark name, benchmark version, and agent configuration
file.

.. code:: bash

   $ scl benchmark run driving_smarts "./baselines/driving_smarts/v0/agent_config.yaml" --auto-install # --auto-install only needs to be used to get dependencies.
   Starting `Driving SMARTS V1` benchmark.
   This is a cleaned up version of the Driving SMARTS benchmark.

       Using `TargetPose` agent action has an applied 28m/s cap for agent motion.
       Using `RelativeTargetPose` agent action, the constraint is inbuilt into the action space.

       For history see: 
           - https://codalab.lisn.upsaclay.fr/competitions/6618
           - https://smarts-project.github.io/archive/2022_nips_driving_smarts/
   Evaluating 1_to_2lane_left_turn_c...
   Evaluating 3lane_merge_multi_agent...
   ...
   Scoring 1_to_2lane_left_turn_c...
   Evaluation complete...

   `Driving SMARTS V0` result:
   - completion: 1
   - humanness: 0.7
   - rules: 0.9
   - time: 0.2
   - overall: 0.504

A particular version of a benchmark can be targeted using a modified
syntax ``benchmark_name==version``:

.. code:: bash

   $ scl benchmark run driving_smarts==0.0 "./baselines/driving_smarts/v0/agent_config.yaml"

Advanced Configuration
----------------------

``--benchmark-listing``
~~~~~~~~~~~~~~~~~~~~~~~

``scl benchmark run``
^^^^^^^^^^^^^^^^^^^^^

The benchmark listing file is used by ``scl benchmark run`` to determine
what benchmarks are currently available. This can be passed using
``--benchmark-listing`` to provide a different list of benchmarks.

.. code:: bash

   $ scl benchmark run --benchmark-listing benchmark_listing.yaml driving_smarts "./baselines/driving_smarts/v0/agent_config.yaml"

WARNING! Since with ``scl benchmark run`` this listing directs to a code
``entrypoint`` do not use this with a listing file from an unknown
source.

``scl benchmark list``
^^^^^^^^^^^^^^^^^^^^^^

This option also appears on ``scl benchmark list`` to examine a listing
file.

.. code:: bash

   $ scl benchmark list --benchmark-listing benchmark_listing.yaml

Listing File
^^^^^^^^^^^^

The listing file is organised as below.

.. code:: yaml

   # smarts/benchmark/benchmark_listing.yaml
   ---
   benchmarks: # The root element (required)
     driving_smarts: # The id of the benchmark for reference
       name: "Driving SMARTS" # The human readable name of the benchmark
       versions: # A list of benchmark versions
         -
           # the version of the benchmark, higher is newer
           version: 0.0
           # the entrypoint for the benchmark, it must have `agent_config`, and `debug_log` as params
           entrypoint: "smarts.benchmark.entrypoints.benchmark_runner_v0.benchmark_from_configs"
           requirements: ["ray<=2.2.0,>2.0"] # requirements to install if `--auto-install`.
           params: # additional values to pass into the entrypoint as named keyword arguments.
             benchmark_config: ${{smarts.benchmark.driving_smarts.v0}}/config.yaml

Resolving module directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmark configuration directory can be dynamically found through
python using an evaluation syntax ``${{}}``. This is experimental and
open to change but the following resolves the python module location in
loaded configuration files:

.. code:: yaml

   somewhere_path: ${{module.to.resolve}}/file.txt # resolves to <path>/module/to/resolve/file.txt

This avoids loading the module into python but resolves to the first
path that matches the module.
