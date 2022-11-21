SMARTS Performance Benchmark
=============================
SMARTS Performance Benchmark is a development tool which is designed for:
- Track the performance of SMARTS simulation for each version
- Test the effects of performance improvements/optimizations

## Setup
- Dump different numbers of actors with different type respectively into 10 secs on a proper map without visualization, and show the mean and standard deviation of time step per second of the corresponding scenarios.
    - n social agents: 1, 10, 20, 50
    - n data replay actors: 1, 10, 20, 50, 200
    - n sumo traffic actors: 1, 10, 20, 50, 200
    - 10 agents to n data replay actors: 1, 10, 20, 50
    - 10 agent to n roads: 1, 10, 20, 50

### Running
Follow the SMARTS setup instruction in the main [README](https://github.com/huawei-noah/SMARTS/). Then run the following command to start the benchmark with one or multiple scenarios, which are provided in the `./smarts/benchmark` folder.

```bash
$ scl benchmark run <scenarios/path> [scenarios/path]
# e.g., scl benchmark run n_sumo_actors/1_actors
# By giving a directory or directories, all scenarios in the directory/directories will be ran
```

