.. _scenario:

Creating Scenarios
------------------

## Domain Specific Language

The `sstudio` DSL has a simple [ontology](./types.py) to help you express your scenario needs in SMARTS. A template looks as follows,

```python
# Custom Map Format
def custom_map_builder(map_spec):
   ...
   return map_object, map_hash
MapSpec(source="path_or_uri", builder_fn=custom_map_builder)

# Traffic
Traffic(
    engine="SUMO",
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
