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

   $ scl benchmark run driving_smarts smarts.zoo:random-relative-target-pose-agent-v0 --auto-install 
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

See available benchmarks
------------------------

The ``scl benchmark list`` command can be used to see the list of available benchmarks.

.. code:: bash

   $ scl benchmark list 
   BENCHMARK_NAME               BENCHMARK_ID             VERSIONS
   - Driving SMARTS:            driving_smarts           0.0 0.1

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
