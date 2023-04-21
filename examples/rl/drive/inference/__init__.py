from contrib_policy.policy import Policy

from smarts.core.agent_interface import RGB, AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register


def entry_point(**kwargs):
    interface = AgentInterface(
        action=ActionSpaceType.RelativeTargetPose,
        drivable_area_grid_map=False,
        lane_positions=True,
        lidar_point_cloud=False,
        occupancy_grid_map=False,
        road_waypoints=False,
        signals=False,
        top_down_rgb=RGB(
            width=128,
            height=128,
            resolution=80 / 128,  # m/pixels
        ),
    )

    agent_params = {
        "top_down_rgb": interface.top_down_rgb,
        "action_space_type": interface.action,
        "num_stack": 3,  # Number of frames to stack as input to policy network.
    }

    return AgentSpec(
        interface=interface,
        agent_builder=Policy,
        agent_params=agent_params,
    )


register(locator="contrib-agent-v0", entry_point=entry_point)
