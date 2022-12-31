.. _scenario:

Creating Scenarios
------------------


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