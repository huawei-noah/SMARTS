.. _benchmark:

Instructions
============

A benchmark is a standard set of rigorous environments which can be used to 
assess and compare the performance of agents built by various researchers. 

:doc:`Agent Zoo </benchmarks/agent_zoo>` contains prebuilt and trained agents 
which could be deployed as reference ego agent in the benchmarks. Feel free to 
mix and match compatible agents and benchmarks.

Run a benchmark
---------------

| Run a particular benchmark by executing 
| ``scl benchmark run <benchmark_name>==<benchmark_version> <agent_locator> --auto-install`` 

The ``--auto-install`` flag is optional and is only needed for the
first time the benchmark is run to install the benchmark's dependencies.

If ``scl benchmark run <benchmark_name> <agent_locator>`` is run without the
benchmark version, then the benchmark's latest version is run by default.

.. code:: bash

   $ scl benchmark run driving_smarts_2022 smarts.zoo:random-relative-target-pose-agent-v0 --auto-install 
   
   <-- Starting `Driving SMARTS 2022` benchmark -->
   
   This is a cleaned up version of the Driving SMARTS benchmark.

       Using `TargetPose` agent action has an applied 28m/s cap for agent motion.
       Using `RelativeTargetPose` agent action, the constraint is inbuilt into the action space.

       For history see: 
           - https://codalab.lisn.upsaclay.fr/competitions/6618
           - https://smarts-project.github.io/archive/2022_nips_driving_smarts/competition/

   Evaluating 1_to_2lane_left_turn_c...
   Evaluating 3lane_merge_multi_agent...
   ...
   Scoring 1_to_2lane_left_turn_c...

   SCORE
   {'overall': 0.424,
    'dist_to_destination': 0.925,
    'humanness': 0.769,
    'rules': 1.0,
    'time': 0.265}
   
   <-- Evaluation complete -->

See available benchmarks
------------------------

The ``scl benchmark list`` command can be used to see the list of available benchmarks.

.. code:: bash

   $ scl benchmark list 
   BENCHMARK_NAME               BENCHMARK_ID             VERSIONS
   - Driving SMARTS 2022:       driving_smarts_2022      0.0 0.1

Custom benchmark listing
------------------------

The ``scl benchmark run`` uses a 
`default benchmark listing <https://github.com/huawei-noah/SMARTS/blob/master/smarts/benchmark/benchmark_listing.yaml>`_ 
file to determine the currently available benchmarks. Alternatively, a custom
benchmark listing file may be supplied as follows.   

.. code:: bash

   $ scl benchmark run --benchmark-listing benchmark_listing.yaml <benchmark_name> <agent_locator>

.. warning::

    Since a listing directs ``scl benchmark run`` to execute an 
    ``entrypoint`` code, do not use this with a listing file from an unknown
    source.

The list of benchmarks from the custom benchmark listing file can be examined as usual.

.. code:: bash

   $ scl benchmark list --benchmark-listing benchmark_listing.yaml

Benchmark listing file
----------------------

The benchmark listing file is organised as below.

.. code:: yaml

   # smarts/benchmark/benchmark_listing.yaml
   ---
   benchmarks: # The root element (required)
     driving_smarts_2022: # The id of the benchmark for reference
       name: "Driving SMARTS 2022" # The human readable name of the benchmark
       versions: # A list of benchmark versions
         -
           # The version of the benchmark, higher is newer
           version: 0.0
           # The entrypoint for the benchmark, it must have `agent_config`, and `debug_log` as params
           entrypoint: "smarts.benchmark.entrypoints.benchmark_runner_v0.benchmark_from_configs"
           requirements: ["ray<=2.2.0,>2.0"] # Requirements to install if `--auto-install`.
           params: # Additional values to pass into the entrypoint as named keyword arguments.
             benchmark_config: ${{smarts.benchmark.driving_smarts.v2022}}/config.yaml

.. note:: 
    
    Resolving module directories.

    The benchmark configuration directory can be dynamically found through
    python using an evaluation syntax ``${{}}``. This is experimental and
    open to change but the following resolves the python module location in
    loaded configuration files:

    .. code:: yaml

        somewhere_path: ${{module.to.resolve}}/file.txt # resolves to <path>/module/to/resolve/file.txt

    This avoids loading the module into python but resolves to the first
    path that matches the module.
