# ROS Driver for SMARTS

This is a catkin workspace for a ROS (v1) node that wraps/drives a SMARTS simulation.

## ROS Installation and Configuration

First, see [wiki.ros.org](http://wiki.ros.org) for instructions about installing and configuring a ROS environment.

### Python version issues

Note that SMARTS uses **python3**, whereas ROS verions `1.*` (kinetic, lunar, melodic, or noetic) were designed for **python2**.

The example node in `src/smarts_ros/scripts/ros_driver.py` was created for the "_kinetic_" ROS distribution and may not work with a ROS `2.*` distribution.
Therefore, you may need to tweak your ROS and/or python environment(s) slightly to get things to work.

The exact tweaks/workarounds to get python3 code running correctly with ROS version `1.*` will depend upon your local setup.
But among other things, you may need to do the following (after the "normal" SMARTS and ROS installation and setups):
```bash
source .venv/bin/activate
pip3 install rospkg catkin_pkg
```

## Setup

Setup your environment:
```bash
source /opt/ros/kinetic/setup.bash
```
```bash
cd examples/ros
catkin_make
catkin_make install
source install/setup.bash
```


## Running

From the main SMARTS repo folder:
```bash
roslaunch smarts_ros ros_driver.launch
```
or:
```bash
rosrun smarts_ros ros_driver.py
```
Or if you prefer (or if required due to the python version issues desribed above):
```bash
python3 exmples/ros/src/smarts_ros/scripts/ros_driver.py
```

These latter 2 may require you to explicitly start `rosmaster` node first
if you don't already have an instance running, like:
```bash
roscore &
```
which will run one in the background.

Alternatively, if you have parameters that you want to override on a regular basis,
create a custom [roslaunch](http://wiki.ros.org/roslaunch) file in your package's launch folder,
like the one in [examples/ros/src/smarts_ros/launch/ros_driver.launch](examples/ros/src/smarts_ros/launch/ros_driver.launch).
And then, if you called it `my_ros_driver.launch`:
```bash
roslaunch smarts_ros launch/my_ros_driver.launch
```
(This approach will automatically start the `rosmaster` node.)


### Parameters and Arguments

The node started by `ros_driver.py` accepts several parameters.  
These can be specified as arguments on the command line when using `rosrun`
or added to a `.launch` file when using `roslaunch`, or set in the 
ROS parameter server.

In addition to the normal arguments that `roslaunch` supplies on
the command line (e.g., `__name`, `__ns`, `__master`, `__ip`, etc.)
the following other (optional) arguments will set the associated
ROS private node parameters if used via `rosrun`:

- `_buffer_size`:  The number of entity messages to buffer to use for smoothing/extrapolation.  Must be a positive integer.  Defaults to `3`.

- `_target_freq`:  The target frequencey in Hz.  If not specified, the node will publish as quickly as SMARTS permits.

- `_time_ratio`:  How many real seconds should a simulation second take.  Must be a positive float.  Default to `1.0`.

- `_headless`:  Controls whether SMARTS should also emit its state to an Envision server.  Defaults to `True`.

- `_seed`:  Seed to use when initializing SMARTS' random number generator(s).  Defaults to `42`.

- `_batch_mode`:  If `True`, the node will stay alive even if SMARTS crashes/dies, waiting for a new `SmartsReset` message on the `SMARTS/reset` topic.  Defaults to `False`.


To specify these via the command line, use syntax like:
```bash
rosrun smarts_ros ros_driver.py _target_freq:=20
```


### Scenarios

Then, when you want to initialize SMARTS on a scenario,
have one of the nodes on the ROS network publish an appropriate `SmartsReset` messsage on the `SMARTS/reset` topic,
after which SMARTS will begin handling messages from the `SMARTS/entities_in` channel.

Or you could manually reset SMARTS from the command line with:
```bash
rostopic pub /SMARTS/reset smarts_ros/SmartsReset '{ reset_with_scenario_path: /full/path/to/scenario }'
```


### SMARTS Info service
If you need to find out basic information about a running SMARTS node,
you can query the SMARTS info service, like:
```python
import rospy
from smarts_ros.srv import SmartsInfo

smarts_info_req = rospy.ServiceProxy("SMARTS/SMARTS_info", SmartsInfo)
smarts_info = smarts_info_req()

# Can now use fields like:
#   smarts_info.version
#   smarts_info.step_count
#   smarts_info.elapsed_time
#   smarts_info.current_scenario_path
```
