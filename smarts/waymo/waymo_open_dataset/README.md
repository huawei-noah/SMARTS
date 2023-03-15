# Waymo Open Dataset

This module is intended to eliminate dependence on the [`waymo_open_dataset`](https://github.com/waymo-research/waymo-open-dataset) package in SMARTS. We only include the `scenario` and `map` protobuf definitions, as well as their compiled Python files. The current Python files were generated using version `3.19.6` of the protobuf compiler. See the below instructions if you encounter any errors related to protobuf versions that require regenerating the Python files.

# Compilation Instructions

To regenerate the `scenario_pb2.py` and `map_pb2.py` files:

1. Install the protobuf compiler: see https://grpc.io/docs/protoc-installation/
2. (Optional) Download the proto files if necessary

```sh
curl https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/waymo_open_dataset/protos/scenario.proto -o waymo_open_dataset/protos/scenario.proto

curl https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/waymo_open_dataset/protos/map.proto -o waymo_open_dataset/protos/map.proto
```

3. Compile the proto files

Run from the base directory of the SMARTS repo:
```sh
protoc -I=. --python_out=. smarts/waymo/waymo_open_dataset/protos/scenario.proto && protoc -I=. --python_out=. smarts/waymo/waymo_open_dataset/protos/map.proto
```
