# ROS Driver for SMARTS

This is a catkin workspace for a ROS (v1) node that wraps/drives a SMARTS simulation.

## ROS Installation and Configuration

First, see [wiki.ros.org](http://wiki.ros.org) for instructions about installing and configuring a ROS environment.

### Python version issues

Note that SMARTS uses **python3**, whereas ROS verions `1.*` (kinetic, lunar, melodic, or noetic) was designed for **python2**.

The example node in `src/src/ros_wrapper.py` was created for the "kinetic" ROS distribution and may not work with a ROS `2.*` distribution.
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
cd examples/ros
catkin_make
source devel/setup.bash
```

Then, if you don't already have a `roscore` instance running, start ROS in the background with:
```bash
roscore &
```

## Running

From the main SMARTS repo folder, for some scenario path `<scenario>`:

```bash
python3 exmples/src/src/ros_wrapper.py <scenerio>
```
(This is prefarable to using `rosrun` or `roslaunch` due to the python verions issues described above.)
