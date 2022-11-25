Performance Benchmark
=====================
A development tool designed to
- Track the performance of each SMARTS version.
- Test the effects of improvements or optimizations made in SMARTS.

## Setup
Dump different number of actors, with different types, on a map without visualization and show the mean & standard deviation of steps per second. Available scenarios:
- n social agents: 1, 10, 20, 50
- n data replay actors: 1, 10, 20, 50, 200
- n sumo traffic actors: 1, 10, 20, 50, 200
- 10 agents to n data replay actors: 1, 10, 20, 50
- 10 agent to n roads: 1, 10, 20, 50

## Run
Run the benchmark with one or multiple scenarios, from the `./smarts/benchmark` folder.

```bash
$ scl benchmark run <scenarios/path> [scenarios/path]
# e.g., scl benchmark run n_sumo_actors/1_actors
```

