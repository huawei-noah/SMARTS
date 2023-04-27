# Change Log
All notable changes to this project will be documented in this file.

This changelog is to adhere to the format given at [keepachangelog](keepachangelog.com/en/1.0.0/)
and should maintain [semantic versioning](semver.org).

All text added must be human-readable.

Copy and pasting the git commit messages is __NOT__ enough.

## [Unreleased]
### Added
- Added an actor capture manager interface, `ActorCaptureManager`, which describes a manager that handles the change of control of actors. Operations in an actor manager step should not cause conflict in the simulation.
- Added a new entry tactic, `IdEntryTactic`, which provides the scenario the ability to select a specific actor for an agent to take over.
- Registered a new `chase-via-points-agent-v0` agent in agent zoo, which can effectively chase via points across different road sections by using the waypoints.
- Added new driving-smarts-v2023 benchmark consisting of new (i) driving-smarts-v2023 env and (ii) platoon-v0 env.
- Added baseline example, consisting of training, inference, and zoo agent registration, for the platooning task in Driving SMARTS 2023.3 benchmark.
- Documented the challenge objective, desired inference code structure, and use of baseline example, for Driving SMARTS 2023.3 benchmark, i.e., platooning task.
- Added a new scenario consisting of merge-exit map, sumo lead vehicle, and traffic, for the vehicle-following task.
- Added a `SensorManager` which manages placing sensors on actors in the simulations.
- The `VehicleState` now has the `bounding_box_points` property to get the vehicle minimum bounding box as a set of points.
- Added engine configuration options for `core:debug`, `core:observation_workers`, and `core:reset_retries`.
- Explained in the docs that agents may spawn at different times in multiagent scenarios.
- Added `RaySensorResolver` as an alternative parallel resolver.
- Added `[ray]` option for `smarts` package. This currently conflicts with `[rllib]`.
- Added engine `observation_workers` configuration which can be used to configure the number of parallel sensor workers: 0 runs the sensors on the local thread, >=1 runs using the multiprocessing backing.
- Added engine `sensor_parallelization` configuration of sensor parallelization backing, options ("mp"|"ray"): "mp" python multiprocessing, "ray" ray worker backing.
- Added engine `reset_retries` configuration engine retries before the simulator will raise an error on reset.
- Introduced new comfort cost function in metric module.
- Introduced new gap-between-vehicles cost function in metric module.
- Added baseline example, consisting of training, inference, and zoo agent registration, for the driving and turning tasks in Driving SMARTS 2023.1 and 2023.2 benchmarks, respectively. It uses RelativeTargetPose action space.
- Documented the challenge objective, desired inference code structure, and use of baseline example, for Driving SMARTS 2023.1 (i.e., basic motion planning) and 2023.2 (i.e, turns) benchmarks.
- Added an env wrapper for constraining the relative target pose action range.
- Added a specialised metric formula module for Driving SMARTS 2023.1 and 2023.2 benchmark.
### Changed
- The trap manager, `TrapManager`, is now a subclass of `ActorCaptureManager`.
- Considering lane-change time ranges between 3s and 6s, assuming a speed of 13.89m/s, the via sensor lane acquisition range was increased from 40m to 80m, for better driving ability.
- The `AgentType.Full` now includes `road_waypoints`, `accelerometer`, and `lane_positions`.
- `ActionSpaceType` has been moved from `controller` to its own file.
- `VehicleState` has been moved to its own file.
- Sensors are no longer added and configured in the `agent_manager`. They are instead managed in the `sensor_manager`.
- Renamed all terminology relating to actor to owner in `VehicleIndex`.
- Renamed all terminology relating to shadow actor to shadower in `VehicleIndex`.
- `Collision` has been moved from `smarts.core.observations` to `smarts.core.vehicle_state`.
- The trap manager, `TrapManager`, is now a subclass of `ActorCaptureManager`.
- Considering lane-change time ranges between 3s and 6s, assuming a speed of 13.89m/s, the via sensor lane acquisition range was increased from 40m to 80m, for better driving ability.
- Modified naming of benchmark used in NeurIPS 2022 from driving-smarts-competition-env to driving-smarts-v2022.
- Social agent actor vehicles are now exactly named the same as the `name` of the actor. 
- Sstudio generated scenario vehicle traffic ids are now shortened.
- ChaseViaPoints zoo agent uses unconstrained path change command, instead of being constrained to [-1, 0, +1] path change commands used previously. 
- Made the metrics module configurable by supplying parameters through a `Params` class.
- Neighborhood vehicles which should be excluded from the `dist_to_obstacles` cost function can be specified through `Params`. This would be useful in certain tasks, like the vehicle-following task where the distance to the lead vehicle should not be included in the computation of the `dist_to_obstacles` cost function.
- Unified the computation of `dist_to_destination` (previously known as `completion`) and `steps` (i.e., time taken) as functions inside the cost functions module, instead of computing them separately in a different module.
- In the metrics module, the records which is the raw metrics data and the scoring which is the formula to compute the final results are now separated to provided greater flexibility for applying metrics to different environments.
- Benchmark listing may specify specialised metric formula for each benchmark.
- Changed `benchmark_runner_v0.py` to only average records across scenarios that share the same environment. Records are not averaged across different environments, because the scoring formula may differ in different environments.
- Renamed GapBetweenVehicles cost to VehicleGap cost in metric module.
- Camera metadata now uses radians instead of degrees.
- The `Panda3d` implementation of `Renderer` has been extracted from the interface and moved to `smarts.p3d`.
- Made all metrics as functions to be minimised, except the overall score which is to be maximised.
- Driving SMARTS 2023.3 benchmark and the metrics module now uses `actor_of_interest_re_filter` from scenario metadata to identify the lead vehicle.
- Included `RelativeTargetPose` action space to the set of allowed action spaces in `platoon-v0` env.
### Deprecated
### Fixed
- Fixed issues related to waypoints in junctions on Argoverse maps. Waypoints will now be generated for all paths leading through the lane(s) the vehicle is on.
- Fixed an issue where Argoverse scenarios with a `Mission` would not run properly.
- `Trip.actor` field is now effective. Previously `actor` had no effect.
- Fixed an issue where building sumo scenarios would sometimes stall.
- `VehicleIndex` no longer segfaults when attempting to `repr()` it.
- Fixed issues related to waypoints in SUMO maps. Waypoints in junctions should now return all possible paths through the junction.
- Fixed CI tests for metrics.
- Fixed an issue where the actor states and vehicle states were not synchronized after simulation vehicle updates resulting in different values from the simulation frame.
- Minor fix in regular expression compilation of `actor_of_interest_re_filter` from scenario metadata.
- Fixed acceleration and jerk computation in comfort metric, by ignoring vehicle position jitters smaller than a threshold.
### Removed
- Removed the deprecated `waymo_browser` utility.
- Removed camera observation `created_at` attribute from metadata to make observation completely reproducible.
### Security

## [1.0.11] # 2023-04-02
### Added
### Changed
- Moved benchmark scenarios into SMARTS/scenarios folder.
- Simplified the scenario loading code in driving_smarts benchmark.
- The `"hiway-v1"` environment now uses `ScenarioOrder` configuration rather than a boolean.
### Deprecated
### Fixed
- Fix case where heading source attribute could be undefined.
- Updated interaction aware motion prediciton zoo agent to work with smarts.
- Edge dividers no longer flow through intersections in Argoverse 2 maps.
### Removed
### Security

## [1.0.10] # 2023-03-27
### Added
- Added vehicle of interest coloring through scenario studio. This lets the scenario color vehicles that match a certain pattern of vehicle id.
- SMARTS now provides `remove_provider` to remove a provider from the simulation. Use carefully.
### Changed
### Deprecated
### Fixed
- Fixed "rl/racing" `numpy` incompatibility.
- Fixed an issue with SUMO maps where waypoints in junctions would not return all possible paths.
- Fixed an issue in Argoverse maps where adjacent lanes would sometimes not be grouped in the same road.
### Removed
### Security

## [1.0.9] # 2023-03-20
### Added
- Added support for the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html#forecasting-link) (see `scenarios/argoverse`)
- Added `waymo_open_dataset` as a module at the SMARTS repo level, to be able to load waymo scenarios without any external packages
### Changed
- Changed the `lanepoint_spacing` setting in `MapSpec` to be non-optional. Lanepoints are now generated lazily when waypoints are used.
### Deprecated
### Fixed
### Removed
- Removed `waymo-open-dataset-tf-2-4-0` package as a dependency
### Security

## [1.0.8] # 2023-03-10
### Added
- Agent manager now has `add_and_emit_social_agent` to generate a new social agent that is immediately in control of a vehicle.
### Changed
- Changed the minimum supported Python version from 3.7 to 3.8
### Deprecated
### Fixed
- Fixed `hiway-v1` environment to use `"render_modes"` instead of `"render.modes"`.
- Fixed an issue with SMARTS where the social vehicles started instantly regardless of what mission start time they were given.
- Missing waypoint paths `'lane_id'`  is now added to the `hiway-v1` formatted observations.
- Engine config utility now properly evaluates `[Ff]alse` as `False` when using a `bool` cast.
### Removed
### Security

## [1.0.7] # 2023-03-04
### Added
- Added objective, scenario description, and trained agent performance, to the Driving Smarts 2022 benchmark documentation.
### Changed
- Unique id suffix is removed from vehicle name while building agent vehicle in `VehicleIndex.build_agent_vehicle()` function. 
### Deprecated
### Fixed
- Missing neighborhood vehicle ids are now added to the `highway-v1` formatted observations.
- Stopped agent providers from removing social agents when they have no actor.
- Using `trip` in sstudio traffic generation no longer causes a durouter error.
- Chassis collision AABB first pass now has an additional `0.05m` tolerance to identify axis aligned collisions that would previously be missed.
- Agent to mission padding warning now occurs when there are less missions than agents rather than when there are the same number of agents as missions.
- Agent manager should no longer de-synchronize vehicle ids with the vehicle index.
### Removed
### Security

## [1.0.6] # 2023-02-26
### Added
- Added a math utility for generating combination groups out of two sequences with unique index use per group. This is intended for use to generate the combinations needed to give a unique agent-mission set per reset.
- Added basic tests for `hiway-v1` resetting and unformatted observations and actions.
- Added `"steps_completed"` to observation formatter.
### Fixed
- Ensured that `hiwayenv.reset` provides unique agent-mission sets per reset.
- Fixed an issue where `sstudio.types.Via` was not hashable.

## [1.0.5] # 2023-02-19
### Added
- Added a zoo agent, named Control-and-Supervised-Learning, from NeurIPS 2022 submission. This zoo agent runs in benchmark `driving_smarts==0.0`.
- Added a zoo agent, named Discrete Soft Actor Critic, from NeurIPS 2022 submission. This zoo agent runs in benchmark `driving_smarts==0.0`.
- Added basic tests for `hiway-v1` resetting and unformatted observations and actions.
- Added `actor_ids` as a provider interface to check the actors that the provider is currently in charge of.
### Changed
- `HiWayEnvV1` derived environments now allow an explicit scenario through `reset(options["scenario"])`.
- `HiWayEnvV1` derived environments now allow an explicit simulation start time through `reset(options["start_time"])`.
- Exposed `smarts` as a property on `HiWayEnvV1`.
- Made the heading input relative to the current heading in `RelativeTargetPose` action space.
### Deprecated
### Fixed
- Issue where a 0 length lane caused `envision` to crash.
- Fixed an issue where `Feature.type_specific_info` was calling a non-existant method.
### Removed
### Security

## [1.0.4] # 2023-02-10
### Added
- Engine configuration utility that uses the following locations to allow configuration of the SMARTS engine. The engine consumes the configuration files from the following locations in the following priority: `./engine.ini`, `~/.smarts/engine.ini`, `$GLOBAL_USER/smarts/engine.ini`, and `${PYTHON_ENV}/lib/${PYTHON_VERSION}/site-packages/smarts/engine.ini`.
- Added map source uri as `map_source` inside of `hiway-v1` reset info to indicate what the current map is on reset.
- Added NGSIM documentation.
- Added a zoo agent, named Interaction-Aware Motion Prediction, from NeurIPS2022 submission. This zoo agent runs in benchmark `driving_smarts==0.0`.
- Added Agent Zoo documentation in ReadTheDocs.
### Changed
- Made changes in the docs to reflect `master` branch as the main development branch.
- Enabled supplying agent locator directly to benchmark runner and removed the need for an intermediary config file. Updated benchmark docs to reflect this.
- Individualised the agent instances in the `benchmark_runner_v0.py`.
- Made driving_smarts_competition_v0 env configurable through supply of `AgentInterface`.
- Observation of driving_smarts_competition_v0 env was fixed to be of type `ObservationOptions.unformatted`.
### Deprecated
### Fixed
- Fixed an exit error that occurs when envision attempts to close down.
- Clarified the actions for `ActionSpaceType.Continuous` and `ActionSpaceType.ActuatorDynamic` in their respective docstrings.
- Excluded from wheel any scenario build files in pattern `smarts/**/build/**/*.xml`.
- Fixed an unintended regression in the metrics.
### Removed
- Removed duplicated `smarts.env.gymnasium.action_conversion` module.
### Security

## [1.0.3] 2023-02-04
### Added
- Added action formatting option to `hiway-v0`.
- Introduced `debug: serial: bool` option to driving smarts benchmark config.
### Changed
- Moved action and observation conversions from `smarts.env.gymnasium.utils` to `smarts.env.utils`.
### Fixed
- Fixed an issue where termination while envision is enabled but not connected would cause a flurry of broken pipe errors.
- Fixed an issue where activating visdom would cause errors.
- Fixed an issue where metrics break down with unformatted observations.
- Fixed an issue where `hiway-v1` would cause an exception when using "unformatted" observations.
- Unformatted actions and observations in `hiway-v0` provide `None` rather than an incorrect space.

## [1.0.2] 2023-01-27
### Added
- The `hiway-v1` environment can now be configured to provide an "unformatted" observation. 
### Changed
- Scenario paths is no longer manually supplied to Envision server while setup. Scenario paths are automatically sent to Envision server from SMARTS during simulation startup phase.
- Updated "hiway-v1" with `gymnasium` action spaces using new `ActionsSpaceFormatter` utility.
### Fixed
- Fixed an issue where a sensor detach call when a bubble vehicle exits a bubble could cause a program crash.
- Fixed issue with "hiway-v0" where "agent_interfaces" was not populated.
- Add missing `distance_travelled` to the `hiway-v1` observations.

## [1.0.1] 2023-01-24
### Fixed
- Fixed issue where Driving SMARTS benchmark only did 2 evaluations per scenario instead of 50.
- Removed unclosed file warnings from benchmark output.

## [1.0.0] 2023-01-22
### Added
- Exposed `.glb` file metadata through the scenario `Scenario.map_glb_metadata` attribute.
- Added single vehicle `Trip` into type. 
- Added new video record ultility using moviepy.
- Added distance check between bubble and vehicle to avoid generating unnecessary cursors.
- Added `ConfigurableZone` for `Zone` object to types which enable users to build bubble by providing coordinates of the polygon.
- Added "SMARTS Performance Diagnostic" development tool for evaluating the simulation performance.
- Added a "All Simulation" button on the header of Envision and made small-windowed simulation(s) in the "All Simulations" page clickable to maximize.
- An env wrapper `Metrics` is introduced to compute agents' performance metrics.
- Extracted `TraciConn` to the SMARTS utilities as a simplified utility to help with connecting to `TraCI`.
- Added `HiWayV1` `gymansium` environment to smarts. This can be referenced through gymnasium as `smarts.env:hiway-v1`.
- Added `scl benchmark run` and `scl benchmark list` for running and listing benchmarks.
- Added the "driving_smarts" benchmark as a feature of the new `scl benchmark` suite.
- Added `smarts.benchmark` module which deals with running benchmarks.
  - Added `smarts.core.entrypoints.benchmark_runner_v0` which is the initial benchmark fully integrated into `smarts`.
- Added documentation with benchmark information.
### Deprecated
### Changed
- Minimum `SUMO` version allowed by `SumoTrafficSimulation` is now `1.10.0`.
- The `ProviderManager` interface now uses a string id for removal of an actor instead of an actor state.
- Renamed many fields of the `smarts.core.agent_interface.AgentInterface` dataclass: `lidar` -> `lidar_point_cloud`, `waypoints` -> `waypoint_paths`, `rgb` -> `top_down_rgb`, `neighborhood_vehicles` -> `neighborhood_vehicle_states`, and `ogm` -> `occupancy_grid_map`.
- Renamed `smarts.core.provider.Provider`'s `action_spaces` to `actions`.
- Moved `VehicleObservation`, `EgoVehicleObservation`, `Observation`, `RoadWaypoints`, `GridMapMetadata`, `TopDownRGB`, `OccupancyGridMap`, `DrivableAreaGridMap`, `ViaPoint`, `Vias`, `SignalObservation`, and `Collision` from `smarts.core.sensors` to `smarts.core.observations`. They are now all typed `NamedTuples`.
- Renamed `GridMapMetadata` field `camera_pos` to `camera_position`.
### Removed
- Removed all of PyMarl contents, including related interface adapter, environments, and tests.
- Removed ray usage example.
- Moved ULTRA from `huawei-noah/SMARTS` to `smarts-project/smarts-project.rl` repository.
- Removed observation_adapter, reward_adapter, and info_adapter, from `hiway_env`.
- Removed `action_space` field from the `smarts.core.agent_interface.AgentInterface` dataclass.
### Fixed
- Updated the RL example `racing` to use `smarts[camera_obs]==0.7.0rc0` and continuous flowing traffic scenario. Simplified the `racing` RL example folder structure.
- Envision "near realtime" mode bugfix
- Corrected an issue where traffic lights in SUMO traffic simulation could be empty and cause a termination of the simulation.
- Fixed an issue where vehicles could cause SMARTS to terminate from being in multiple providers.
- Fixed an issue where `sumo_traffic_simulation` would disconnect on a non-terminal exception.
- SMARTS now aggressively attempts to connect to a SUMO process as long as the SUMO process remains alive.
- SUMO traffic simulation `route_for_vehicle` had semantic errors and now works again.
- SUMO is now supported up to version `1.15.0`. Versions of SUMO `1.13.0` and onward are forced to reset rather than reload because of errors with hot resetting versions starting with `1.13.0`. 
### Security

## [0.7.0rc0] 2022-10-31
### Added
- Added a basic background traffic ("social vehicle") provider as an alternative to the SUMO traffic simulator.  This can be selected using the new `"engine"` argument to `Traffic` in Scenario Studio.
- Added a `multi-scenario-v0` environment which can build any of the following scenario, namely, `1_to_2lane_left_turn_c`, `1_to_2lane_left_turn_t`, `3lane_merge_multi_agent`, `3lane_merge_single_agent`, `3lane_cruise_multi_agent`, `3lane_cruise_single_agent`, `3lane_cut_in`, and `3lane_overtake`. Additional scenarios can also be built by supplying the paths to the scenario directories.
- Added ego's mission details into the `FormatObs` wrapper.
- Added `SmartsLaneChangingModel` and `SmartsJunctionModel` to types available for use with the new smarts traffic engine within Scenario Studio.
- Added option to `AgentInterface` to include traffic signals (lights) in `EgoVehicleObservation` objects.
- Added the ability to hover over vehicles and roadmap elements in Envision to see debug info.

### Deprecated
- Deprecated a few things related to traffic in the `Scenario` class, including the `route` argument to the `Scenario` initializer, the `route`, `route_filepath` and `route_files_enabled` properties, and the `discover_routes()` static method.  In general, the notion of "route" (singular) here is being replaced with "`traffic_specs`" (plural) that allow for specifying traffic controlled by the SMARTS engine as well as Sumo.
- `waymo_browser.py` has been deprecated in favour of the scl waymo command line tools.

### Changed
- Add `lane_offset` to `Waypoint` class and `lane_postion` to both `EgoVehicleObservation` and `VehicleObservation` classes to expose the reference-line (a.k.a. Frenet) coordinate system.
- Traffic history vehicles can now be hijacked within a bubble.  They will be relinquished to the SMARTS background traffic provider upon exiting the bubble.
- Changed the name of the `Imitation` action space to `Direct`.
- Removed `endless_traffic` option from `SumoTrafficSimulator` and instead added `repeat_route` to `Flow` type in Scenario Studio.
- Renamed `examples/observation_collection_for_imitation_learning.py` to `examples/traffic_histories_to_observations.py`.
- Renamed `examples/history_vehicles_replacement_for_imitation_learning.py` to `examples/traffic_histories_vehicle_replacement.py`.
- `SumoTrafficSimulation` will now try to hand-off the vehicles it controls to the new SMARTS background traffic provider by default if the Sumo provider crashes.
- SMARTS now gives an error about a suspected lack of junction edges in sumo maps on loading of them.
- Scenario build artifacts are now cached and built incrementally, meaning that subsequent builds (without the `clean` option) will only build the artifacts that depend on the changed DSL objects
- All build artifacts are now in a local `build/` directory in each scenario's directory
- The `allow_offset_map` option has been removed. This must now be set in a `MapSpec` object in the scenario.py if this option is needed
- All scenarios must have a `scenario.py`, and must call `gen_scenario()`, rather than the individual `gen_` functions, which are now private

### Removed
- Removed support for deprecated json-based and YAML formats for traffic histories.
- Removed time and distance to collision values from `FormatObs` wrapper as the current implementation's correctness was in doubt.

### Fixed
- Fixed bug where `yaw_rate` was always reported as 0.0 (Issue #1481).
- Modified `FrameStack` wrapper to support agents which start at a later time in the simulation.
- Truncated all waypoint paths returned by `FormatObs` wrapper to be of the same length. Previously, variable waypoint-path lengths caused inhomogenous shape error in numpy array.
- Fixed a bug where traffic providers would leak across instances due to the ~~(awful design decision of python)~~ reference types defaults in arguments sharing across instances.
- Fixed minor bugs causing some Waymo maps not to load properly.
- Fixed a bug where `Vehicle.bounding_box` was mirrored over Y causing on shoulder events to fire inappropriately.
- Fixed an issue where the ego and neighbour vehicle observation was returning `None` for the nearby `lane_id`, `lane_index`, and `road_id`. These now default to constants `off_lane`, `-1`, and `off_road` respectively.
- Fixed a bug where bubble agents would stick around and to try to get observations even after being disassociated from a vehicle.
- Fixed a bug with the `TripMeterSensor` that was not using a unit direction vector to calculate trip distance against current route.
- Fixed issues with Envision. The playback bar and realtime mode now work as expected.
- Fixed a bug where traffic history vehicles would not get traffic signal observations
- Fixed a bug where envision would not work in some versions of python due to nuances of `importlib.resource.path()`.
- Fixed an issue with incorrect vehicle sizes in Envision.

## [0.6.1] 2022-08-02
### Added
- Added standard intersection environment, `intersection-v0`, for reinforcement learning where agents have to make an unprotected left turn in the presence of traffic.
- Added an online RL example for solving the `intersection-v0` environment, using PPO algorithm from Stable Baselines3 library. An accompanying Colab example is also provided.

### Changed
- Updated license to 2022 version.
- SMARTS reset now has a start time option which will skip simulation.
- Since `gym.Space` does not support dataclass, `StdObs` type is changed from a dataclass to a dictionary.

### Removed
- Old Stable Baselines3 based example is removed in favour of the new online RL example developed using Stable Baselines3 library.

### Fixed
- Additional case added for avoiding off-route if merging early into a lane.
- Unpack utility now unpacks dataclass attributes.
- Trap manager now uses elapsed sim time rather than step delta to associate with time.

## [0.6.1rc1] 2022-04-18
### Added
- Added example scenario for importing the NGSIM `peachtree` dataset.
- Added example scenario for importing the INTERACTION `merging` dataset
### Deprecated
- Using `.yml` files to specify traffic history datasets have been deprecated in favor of using `sstudio.types.TrafficHistoryDataset` objects.
### Fixed
- Gracefully handle `EndlessGoal` missions in the MaRL benchmark. Relative goal distance with `EndlessGoal` will be now always be 0.
- Restore `rl-agent` to working order. Version incremented to `1.1.0`.
- Update `rl-agent` wheel.
- Do not auto-shift maps for a scenario that has traffic history.
- Fixed Issue #1321 such that numpy's `sliding_window_view()` is no longer needed for NGSIM traffic histories.
- Fixed NGSIM traffic history import bugs (see Issues #1354 and #1402).

## [0.6.1rc0] 2022-04-16
### Added
- Added `smarts/waymo/waymo_browser.py`, a text-based utility to explore and export scenarios from the Waymo Motion dataset to SMARTS scenarios. 
- Added `get_vehicle_start_time()` method for scenarios with traffic history data.  See Issue #1210.
- Added `sb3` reinforcement-learning example. An ego agent is trained using PPO algorithm from Stable Baselines3 library, to drive as far and as fast as possible in heavy traffic, without colliding or going off-road.
- Added `FormatObs` wrapper which converts SMARTS observations to gym-compliant RL-friendly vectorized observations and returns `StdObs`.
- Added `Pose.as_position2d()` method which converts the pose to an [x,y] position array.
- Added `EventConfiguration` dataclass in the agent interface to allow users to configure the conditions in which events are triggered
- Extended the `RoadMap` API to support `Waymo` map format in `smarts/core/waymo_map.py`.
- Added scenarios for "importing" the i80 and us101 NGSIM trajectory history datasets
- Added an observation adapter that makes the observation ego-centric: `smarts.core.utils.ego_centric_observation_adapter`.
- Added math utility `world_position_from_ego_frame` which allows converting from an ego frame to world frame.
- Added math utility `wrap_value` which constrains a float between a `min` and `max` by wrapping around every time the value exceeds `max` or falls below `min`.
- Added ego-centric adapter utility `smarts.core.utils.adapters.ego_centric_adapters.get_egocentric_adapters(action_space)` which provides an ego-centric pair of observation and action adapters that are used together to provide an ego-centric interface.
### Changed
- If more than one qualifying map file exists in a the `map_spec.source` folder, `get_road_map()` in `default_map_builder.py` will prefer to return the default files (`map.net.xml` or `map.xodr`) if they exist.
- Moved the `smarts_ros` ROS node from the `examples` area into the `smarts.ros` module so that it can be distributed with SMARTS packages.
- Use `Process` to replace `Thread` to speed up the `scl scenario build-all --clean <scenario_dir>` runtime.
- Modified the repository's front page to be more informative and better organised.
- Added an option to `Scenario.scenario_variations()` to make the iterator not yield a cycle.
### Deprecated
- Moved the `AgentSpec` class out of `smarts.core.agent` to `smarts.zoo.agent_spec`.
### Fixed
- Fixed a secondary exception that the `SumoTrafficSimulation` will throw when attempting to close a TraCI connection that is closed by an error.
- Ensure that `smarts.core.coordinates.Pose` attribute `position` is an [x, y, z] numpy array, and attribute `orientation` is a quaternion length 4 numpy array. 
- Update social vehicle pose in Bullet when no active agents are present.
- Fix suppression of `stderr` and `stdout` on `ipython` platforms via `suppress_output(..)`.
### Removed
- Removed the unconditional import of `Renderer` from `smarts/core/vehicle.py` to make `Panda3D` optional dependency regression. See Issue #1310.
### Security

## [0.6.0] 2022-03-28
### Added
- Added `get_vehicle_start_time()` method for scenarios with traffic history data.  See Issue #1210.
- Added `sb3` reinforcement-learning example. An ego agent is trained using PPO algorithm from Stable Baselines3 library, to drive as far and as fast as possible in heavy traffic, without colliding or going off-road.
- Added `FormatObs` wrapper which converts SMARTS observations to gym-compliant RL-friendly vectorized observations and returns `StdObs`.
- Added `Pose.as_position2d()` method which converts the pose to an [x,y] position array.
- Added `EventConfiguration` dataclass in the agent interface to allow users to configure the conditions in which events are triggered
- Added scenarios for "importing" the i80 and us101 NGSIM trajectory history datasets
### Changed
- If more than one qualifying map file exists in a the `map_spec.source` folder, `get_road_map()` in `default_map_builder.py` will prefer to return the default files (`map.net.xml` or `map.xodr`) if they exist.
- Moved the `smarts_ros` ROS node from the `examples` area into the `smarts.ros` module so that it can be distributed with SMARTS packages.
- Use `Process` to replace `Thread` to speed up the `scl scenario build-all --clean <scenario_dir>` runtime.
- Modified the repository's front page to be more informative and better organised.
### Deprecated
- Moved the `AgentSpec` class out of `smarts.core.agent` to `smarts.zoo.agent_spec`.
### Fixed
- Fixed a secondary exception that the `SumoTrafficSimulation` will throw when attempting to close a TraCI connection that is closed by an error.
- Ensure that `smarts.core.coordinates.Pose` attribute `position` is an [x, y, z] numpy array, and attribute `orientation` is a quaternion length 4 numpy array. 
- Update social vehicle pose in Bullet when no active agents are present.
- Document missing action space type `ActionSpaceType.TargetPose`.
### Removed
- Removed the unconditional import of `Renderer` from `smarts/core/vehicle.py` to make `Panda3D` optional dependency regression. See Issue #1310.
### Security


## [0.5.1.post1] 2022-03-11
### Fixed
- Fixed an issue involving relative imports in `examples/rllib/rllib.py`.
- Fixed an issue with uncapped `opencv` causing an error within `ray.rllib`.
- Fixed a longstanding issue that did not allow camera observations unless you had windowing.

## [0.5.1] 2022-01-25
### Added
- Added `get_vehicle_start_time()` method for scenarios with traffic history data.  See Issue #1210.
### Changed
- If more than one qualifying map file exists in a the `map_spec.source` folder, `get_road_map()` in `default_map_builder.py` will prefer to return the default files (`map.net.xml` or `map.xodr`) if they exist.
- Moved the `smarts_ros` ROS node from the `examples` area into the `smarts.ros` module so that it can be distributed with SMARTS packages.
- Use `Process` to replace `Thread` to speed up the `scl scenario build-all --clean <scenario_dir>` runtime.
### Deprecated
### Fixed
- Fixed a secondary exception that the `SumoTrafficSimulation` will throw when attempting to close a TraCI connection that is closed by an error.
### Removed
### Security

## [0.5.0] - 2022-01-07
### Added
- Added Minimum FrameRate tests to measure the fps for `smart.step()` method. See Issue #455.
- Added a ROS wrapper/driver example to wrap SMARTS in a ROS (v1) node.
- Added the ability to pass an optional `time_delta_since_last_step` to SMARTS' `step()` function
  to support variable timesteps for co-simulation.
- Added `step_count` and `elapsed_sim_time` to the `Observation` class.  See PR #974 and Issues #884 and #918.
- Added `dt` to `Observation` class to inform users of the observations of the variable timestep.
- Added the ability to externally update SMARTS state via a new privileged-access `ExternalProvider`.
- Allow specifying "-latest" as a version suffix for zoo locator strings.
- Added Base CI and dependencies requirement tests for the "darwin" platform (MacOS).
- Extended Imitation Learning codebase to allow importing traffic histories from the Waymo motion dataset and replay in a SMARTS simulation. See PR #1060.
- Added options for dealing with noise when inferring headings while importing traffic history data.  See PR #1219.
- Added `ros` extension rule to `setup.py`.
- Added a script to allow users to hijack history vehicles dynamically through a trigger event. See PR #1088.
- Added a `-y` option to `utils/setup/install_deps.sh` to accept installation by default. See issue #1081.
- Added `ParallelEnv` class and a corresponding example to simulate multiple SMARTS environments in parallel, with synchronous or asynchronous episodes.
- Added `smarts.core.utils.import_utils` to help with the dynamic import of modules.
- Added `single_agent` env wrapper and unit test. The wrapper converts a single-agent SMARTS environment's step and reset output to be compliant with gym spaces.
- Added `rgb_image` env wrapper and unit test. The wrapper filters SMARTS environment observation and returns only top-down RGB image as observation.
- Extended the `RoadMap` API to support `OpenDRIVE` map format in `smarts/core/opendrive_road_network.py`. Added 3 new scenarios with `OpenDRIVE` maps. See PR #1186.
- Added a "ReplayAgent" wrapper to allow users to rerun an agent previously run by saving its configurations and inputs. See Issue #971.
- Added `smarts.core.provider.ProviderRecoveryFlags` as flags to determine how `SMARTS` should handle failures in providers. They are as follows:
  - `NOT_REQUIRED`: Not needed for the current step. Error causes skip of provider if it should recover but cannot or should not recover.
  - `EPISODE_REQUIRED`: Needed for the current episode. Results in episode ending if it should recover but cannot or should not recover.
  - `EXPERIMENT_REQUIRED`: Needed for the experiment. Results in exception if it should recover but cannot or should not recover.
  - `ATTEMPT_RECOVERY`: Provider should attempt to recover from the exception or disconnection.
- Added recovery options for providers in `smarts.core.provider.Provider`. These include:
  - Add `recover()` method to providers to attempt to recover from errors and disconnection.
  - Add `connected` property to providers to check if the provider is still connected.
- Added recovery options to `smarts.core.smarts.SMARTS.add_provider()`
  - Add `recovery_flags` argument to configure the recovery options if the provider disconnects or throws an exception.
- Added `driving_in_traffic` reinforcement learning example. An ego agent is trained using DreamerV2 to drive as far and as fast as possible in heavy traffic, without colliding or going off-road.
- Added `smarts.core.smarts.SMARTSDestroyedError` which describes use of a destroyed `SMARTS` instance.
### Changed
- `test-requirements` github action job renamed to `check-requirements-change` and only checks for requirements changes without failing.
- Moved examples tests to `examples` and used relative imports to fix a module collision with `aiohttp`'s `examples` module.
- Made changes to log sections of the scenario step in `smarts.py` to help evaluate smarts performance problems. See Issue #661.
- Introducted `RoadMap` class to abstract away from `SumoRoadNetwork`
  and allow for (eventually) supporting other map formats.  See Issue #830 and PR #1048.
  This had multiple cascading ripple effects (especially on Waypoint generation and caching,
  Missions/Plans/Routes and road/lane-related sensors).  These include:
    - Removed the `AgentBehavior` class and the `agent_behavior` parameter to `AgentInterface`.
    - Moved the definition of `Waypoint` from `smarts.core.mission_planner` to `smarts.core.road_map`.
    - Moved the definition of `Mission` and `Goal` classes from `smarts.core.scenario` to `smarts.core.plan`.
    - Added `MapSpec` to the SStudio DSL types and introduced a simple builder pattern for creating `RoadMap` objects.
- Changed the type hint for `EgoVehicleObservation`: it returns a numpy array (and always has).
- Raised a warning message for building scenarios without `map.net.xml` file. See PR #1161.
- Updated `smarts/env/hiway_env.py` to support `OpenDRIVE` maps so that the `SMARTS` object is instantiated without the `SUMO` traffic provider and social agents. See PR #1215.
- Public `SMARTS` methods will throw `smarts.core.smarts.SMARTSDestroyedError` if `SMARTS.destroy()` has previously been called on the `SMARTS` instance.
### Fixed
- Fix lane vector for the unique cases of lane offset >= lane's length. See PR #1173.
- Logic fixes to the `_snap_internal_holes` and `_snap_external_holes` methods in `smarts.core.sumo_road_network.py` for crude geometry holes of sumo road map. Re-adjusted the entry position of vehicles in `smarts.sstudio.genhistories.py` to avoid false positive events. See PR #992.
- Prevent `test_notebook.ipynb` cells from timing out by increasing time to unlimited using `/metadata/execution/timeout=65536` within the notebook for regular uses, and `pytest` call with `--nb-exec-timeout 65536` option for tests. See for more details: "https://jupyterbook.org/content/execute.html#setting-execution-timeout" and "https://pytest-notebook.readthedocs.io/en/latest/user_guide/tutorial_intro.html#pytest-fixture".
- Stop `multiprocessing.queues.Queue` from throwing an error by importing `multiprocessing.queues` in `envision/utils/multiprocessing_queue.py`.
- Prevent vehicle insertion on top of ignored social vehicles when the `TrapManager` defaults to emitting a vehicle for the ego to control. See PR #1043
- Prevent `TrapManager`from trapping vehicles in Bubble airlocks.  See Issue #1064.
- Social-agent-buffer is instantiated only if the scenario requires social agents
- Mapped Polygon object output of Route.geometry() to sequence of coordinates.
- Updated deprecated Shapely functionality.
- Fixed the type of `position` (pose) fields emitted to envision to match the existing type hints of `tuple`.
- Properly detect whether waypoint is present in mission route, while computing distance travelled by agents with missions in TripMeterSensor.
- Fixed `test_notebook` timeout by setting `pytest --nb-exec-timeout 65536`.
### Deprecated
- The `timestep_sec` property of SMARTS is being deprecated in favor of `fixed_timesep_sec`
  for clarity since we are adding the ability to have variable time steps.
### Removed
- Remove `ray_multi_instance` example when running `make sanity-test`
- Removed deprecated fields from `AgentSpec`:  `policy_builder`, `policy_params`, and `perform_self_test`.
- Removed deprecated class `AgentPolicy` from `agent.py`.
- Removed `route_waypoints` attribute from `smarts.core.sensors.RoadWaypoints`.

## [0.4.18] - 2021-07-22
### Added
- Dockerfile for headless machines.
- Singularity definition file and instructions to build/run singularity containers.
- Support multiple outgoing edges from SUMO maps.
- Added a Cross RL Social Agent in `zoo/policies` as a concrete training examples. See PR #700.
- Made `Ray` and its module `Ray[rllib]` optional as a requirement/dependency to setup SMARTS. See Issue #917.
### Fixed
- Suppress messages in docker containers from missing `/dev/input` folder.
- When code runs on headless machine, panda3d will fallback to using `p3headlessgl` option to render images without requiring X11.
- Fix the case where mapping a blank repository to the docker container `/src` directory via `-v $SMARTS_REPO/src` as directed in the `README` will cause `scl` and other commands to not work.
- Fix case where multiple outgoing edges could cause non-determinism.

## [0.4.17] - 2021-07-02
### Added
- Added `ActionSpace.Imitation` and a controller to support it.  See Issue #844.
- Added a `TraverseGoal` goal for imitation learning agents.  See Issue #848.
- Added `README_pypi.md` to update to the general user installation PyPI instructions. See Issue #828.
- Added a new utility experiment file `cli/run.py` to replace the context given by `supervisord.conf`. See PR #911.
- Added `scl zoo install` command to install zoo policy agents at the specified paths. See Issue #603.
- Added a `FrameStack` wrapper which returns stacked observations for each agent.
### Changed
- `history_vehicles_replacement_for_imitation_learning.py` now uses new Imitation action space. See Issue #844.
- Updated and removed some package versions to ensure that Python3.8 is supported by SMARTS. See issue #266.
- Refactored `Waypoints` into `LanePoints` (static, map-based) and `Waypoints` (dynamic). See Issue #829.
- Vehicles with a `BoxChassis` can now use an `AccelerometerSensor` too.
- When importing NGSIM history data, vehicle speeds are recomputed.
- Allow custom sizes for agent vehicles in history traffic missions.
- Refactored the top level of the SMARTS module to make it easier to navigate the project and understand its structure. See issue #776.
- Made Panda3D and its modules optional as a requirement/dependencies to setup SMARTS. See Issue #883.
- Updated the `Tensorflow` version to `2.2.1` for rl-agent and bump up its version to `1.0`. See Issue #211.
- Made `Ray` and its module `Ray[rllib]` optional as a requirement/dependency to setup SMARTS. See Issue #917.
- Added an error if a `SMARTS` instance reaches program exit without a manual `del` of the instance or a call to `SMARTS.destroy()`.
### Fixed
- Allow for non-dynamic action spaces to have action controllers.  See PR #854.
- Fix a minor bug in `sensors.py` which triggered `wrong_way` event when the vehicle goes into an intersection. See Issue #846.
- Limited the number of workers SMARTS will use to establish remote agents so as to lower memory footprint.
- Patched a restart of SUMO every 50 resets to avoid rampant memory growth.
- Fix bugs in `AccelerometerSensor`.  See PR #878.
- Ensure that `yaw_rate` is always a scalar in `EgoVehicleObservation`.
- Fix the internal holes created at sharp turns due to crude map geometry. See issue #900.
- Fixed an args count error caused by `websocket.on_close()` sending a variable number of args.
- Fixed the multi-instance display of `envision`. See Issue #784.
- Caught abrupt terminate signals, in order to shutdown zoo manager and zoo workers.
- Include tire model in package by moving `tire_parameters.yaml` from `./examples/tools` to `./smarts/core/models`. See Issue #1140
- Fixed an issue where `SMARTS.destroy()` would still cause `SMARTS.__del__()` to throw an error at program exit.
### Removed
- Removed `pview` from `make` as it refers to `.egg` file artifacts that we no longer keep around.
- Removed `supervisord.conf` and `supervisor` from dependencies and requirements. See Issue #802.

## [0.4.16] - 2021-05-11
### Added
- Added `sanity-test` script and asked new users to run `sanity-test` instead of `make test` to ease the setup
process
- Added `on_shoulder` as part of events in observation returned from each step of simulation
- Added description of map creation and how to modify the map to allow users to create their own traffic routes in docs
- Added reference to SMARTS paper in front page of docs
- Only create `Renderer` on demand if vehicles are using camera-based sensors. See issue #725.
- Added glb models for pedestrians and motorcycles
- Added `near realtime` mode and `uncapped` mode in Envision
- Added `--allow-offset-map` option for `scl scenario build` to prevent auto-shifting of Sumo road networks
- Added options in `DoneCriteria` to trigger ego agent to be done based on other agent's done situation
### Changed
- Refactored SMARTS class to not inherit from Panda3D's ShowBase; it's aggregated instead. See issue #597.
- Updated imitation learning examples.
### Fixed
- Fixed the bug of events such as off_road not registering in observation when off_road is set to false in DoneCriteria
- Fixed Sumo road network offset bug for shifted maps.  See issue #716.
- Fixed traffic generation offset bug for shifted maps.  See issue #790.
- Fixed bugs in traffic history and changed interface to it.  See issue #732.
- Update `ego_open_agent` to use the package instead of the zoo directory version.
- Quieted error logs generated by failed Envision connections as well as noisy pybullet log messages.  See issue #819.
- Removed all coverage files created during make test. See issue #826.
- Removed scenarios and examples modules from pip installation. See issue #833.

## [0.4.15] - 2021-03-18
### Added
- This CHANGELOG as a change log to help keep track of changes in the SMARTS project that can get easily lost.
- Hosted Documentation on `readthedocs` and pointed to the smarts paper and useful parts of the documentation in the README.
- Running imitation learning will now create a cached `history_mission.pkl` file in scenario folder that stores
the missions for all agents.
- Added ijson as a dependency.
- Added `cached-property` as a dependency.
### Changed
- Lowered CPU cost of waypoint generation. This will result in a small increase in memory usage.
- Set the number of processes used in `make test` to ignore 2 CPUs if possible.
- Use the dummy OpEn agent (open-agent version 0.0.0) for all examples.
- Improved performance by removing unused traffic light functionality.
- Limit the memory use of traffic histories by incrementally loading the traffic history file with a worker process.
### Fixed
- In order to avoid precision issues in our coordinates with big floating point numbers, we now initially shift road networks (maps) that are offset back to the origin using [netconvert](https://sumo.dlr.de/docs/netconvert.html). We adapt Sumo vehicle positions to take this into account to allow Sumo to continue using the original coordinate system.  See Issue #325. This fix will require all scenarios to be rebuilt (`scl scenario build-all --clean ./scenarios`).
- Cleanly close down the traffic history provider thread. See PR #665.
- Improved the disposal of a SMARTS instance. See issue #378.
- Envision now resumes from current frame after un-pausing.
- Skipped generation of cut-in waypoints if they are further off-road than SMARTS currently supports to avoid process crash.
- Fix envision error 15 by cleanly shutting down the envision worker process.

## [Format] - 2021-03-12
### Added
– Describe any new features that have been added since the last version was released.
### Changed
– Note any changes to the software’s existing functionality.
### Deprecated
– Note any features that were once stable but are no longer and have thus been scheduled for removal.
### Fixed
– List any bugs or errors that have been fixed in a change.
### Removed
– Note any features that have been deleted and removed from the software.
### Security
– Invite users to upgrade and avoid fixed software vulnerabilities.
