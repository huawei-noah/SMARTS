import os
import subprocess
import sys

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "setup_node":
        # Note:  use of this script may require installing the SMARTS package with the "[ros]" extensions.
        mod_path = os.path.dirname(__file__)
        source_ros = ""
        if (
            os.environ.get("ROS_VERSION", 0) != 1
            or os.environ.get("ROS_DISTRO", "a")[0] < "k"
        ):
            ros_base = "/opt/ros"
            for distro in ["noetic", "melodic", "lunar", "kinetic"]:
                ros_distro = os.path.join(ros_base, distro)
                if os.path.isdir(ros_distro):
                    break
            else:
                print(
                    "cannot find appropriate ROS v1 distribution.  SMARTS requires kinetic or newer."
                )
                sys.exit()
            source_ros = f". {ros_distro}/setup.sh && "
        print("setting up ROS node for SMARTS...")
        # need to override path because this is likely to be being run in a venv,
        # but catkin_make for ROS v1 distro should be done with python=python2 not python3.
        default_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        subprocess.check_call(
            f"{source_ros}catkin_make install",
            shell=True,
            cwd=mod_path,
            env={"PATH": default_path},
        )
        print(f"\nnow run:  source {mod_path}/install/setup.bash")
