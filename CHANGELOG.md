# Change Log
All notable changes to this project will be documented in this file.

This changelog is to adhere to the format given at [keepachangelog](keepachangelog.com/en/1.0.0/)
and should maintain [semantic versioning](semver.org).

All text added must be human-readable.

Copy and pasting the git commit messages is __NOT__ enough.

## [Unreleased]
### Added
- Added a basic background traffic ("social vehicle") provider as an alternative to the SUMO traffic simulator.  This can be selected using the new `"engine"` argument to `Traffic` in Scenario Studio.
- Added a `multi-scenario-v0` environment which can build any of the following scenario, namely, `1_to_2lane_left_turn_c`, `1_to_2lane_left_turn_t`, `3lane_merge_multi_agent`, `3lane_merge_single_agent`, `3lane_cruise_multi_agent`, `3lane_cruise_single_agent`, `3lane_cut_in`, and `3lane_overtake`. Additional scenarios can also be built by supplying the paths to the scenario directories.
- Added ego's mission details into the `FormatObs` wrapper.
- Added `SmartsLaneChangingModel` and `SmartsJunctionModel` to types available for use with the new smarts traffic engine within Scenario Studio.
- Added option to `AgentInterface` to include traffic signals (lights) in `EgoVehicleObservation` objects.
- Added SMARTS competition code, consisting of Codalab, training, submission, evaluation, and scenario files.

### Deprecated
- Deprecated a few things related to traffic in the `Scenario` class, including the `route` argument to the `Scenario` initializer, the `route`, `route_filepath` and `route_files_enabled` properties, and the `discover_routes()` static method.  In general, the notion of "route" (singular) here is being replaced with "`traffic_specs`" (plural) that allow for specifying traffic controlled by the SMARTS engine as well as Sumo.

### Changed
- Add `lane_offset` to `Waypoint` class and `lane_postion` to both `EgoVehicleObservation` and `VehicleObservation` classes to expose the reference-line (a.k.a. Frenet) coordinate system.
- Traffic history vehicles can now be hijacked within a bubble.  They will be relinquished to the SMARTS background traffic provider upon exiting the bubble.
- Changed the name of the `Imitation` action space to `Direct`.
- Removed `endless_traffic` option from `SumoTrafficSimulator` and instead added `repeat_route` to `Flow` type in Scenario Studio.
- Renamed `examples/observation_collection_for_imitation_learning.py` to `examples/traffic_histories_to_observations.py`.
- Renamed `examples/history_vehicles_replacement_for_imitation_learning.py` to `examples/traffic_histories_vehicle_replacement.py`.
- `SumoTrafficSimulation` will now try to hand-off the vehicles it controls to the new SMARTS background traffic provider by default if the Sumo provider crashes.

### Removed
- Removed support for deprecated json-based and YAML formats for traffic histories.
- Removed time and distance to collision values from `FormatObs` wrapper as the current implementation's correctness was in doubt.

### Fixed
- Fixed bug where `yaw_rate` was always reported as 0.0 (Issue #1481).
- Modified `FrameStack` wrapper to support agents which start at a later time in the simulation.
- Truncated all waypoint paths returned by `FormatObs` wrapper to be of the same length. Previously, variable waypoint-path lengths caused inhomogenous shape error in numpy array.
- Fixed a bug where traffic providers would leak across instances due to the ~~(awful design decision of python)~~ reference types defaults in arguments sharing across instances.
- Fixed minor bugs causing some Waymo maps not to load properly.
- Fixed bug where `Vehicle.bounding_box` was mirrored over Y causing on shoulder events to fire inappropriately.
- Fixed issue where the ego and neighbour vehicle observation was returning `None` for the nearby `lane_id`, `lane_index`, and `road_id`. These now default to constants `off_lane`, `-1`, and `off_road` respectively.

## [0.6.1]
### Added
- Added standard intersection environment, `intersection-v0`, for reinforcement learning where agents have to make an uprotected left turn in the presence of traffic.
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

## [0.6.1rc1] 15-04-18
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

## [0.6.1rc0] 15-04-16
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
– Note any features that were once stable but are no longer and have thus been removed.
### Fixed
– List any bugs or errors that have been fixed in a change.
### Removed
– Note any features that have been deleted and removed from the software.
### Security
– Invite users to upgrade and avoid fixed software vulnerabilities.
