# Scenario Studio

Scenario Studio is a tool in the SMARTS platform to build and reuse scenarios. At the most basic level scenarios combine a map with a traffic description.

## Creating Scenarios

Scenario creation is done through an asset pipeline configured through `make`. This repo ships with a variety of scenarios such as `intersection`, `loop`, and `figure_eight`. However, creating your own scenarios is easy, and you're encouraged to do so. The workflow is,

1. Create a directory for your new scenario (can put it under [`SMARTS/scenarios`](../scenarios))
2. Design the map in [netedit](https://sumo.dlr.de/docs/NETEDIT.html) and save in your new directory
3. If you want social vehicle traffic or agent missions define a [traffic spec](#Traffic-DSL) in a `scenario.py` file under your scenario directory
4. Build the scenario by running,
    ```bash
    # Assuming your scenario is under SMARTS/scenarios, and you're CWD=SMARTS/
    make build-scenario scenario=scenarios/<your_scenario_dir>
    ```

## Domain Specific Language

The `sstudio` DSL has a simple [ontology](./types.py) to help you express your scenario needs in SMARTS. A template looks as follows,

```python
# Traffic
Traffic(
    flows=[
        Flow(
            route=Route(begin=("edge-A", 0, "base"), end=("edge-B", 0, "max")),
            rate=100,
            actors={TrafficActor(name="car"): 1.0},
        ),
        ...
    ]
)

# Missions
Mission(Route(begin=("edge-C", 0, "base"), end=("edge-D", 0, "max")))
```

Additionally, you can look at the provided scenarios to see how they define their traffic.

## Nota Bene

The "correctness" of traffic and missions is partially your responsibility. Specifically, ensuring that the start positions of ego vehicle mission routes and social vehicle traffic routes don't overlap is not handled by `sstudio`. If they were to overlap a collision will be immediately detected and the episode will end.
