from contrib_policy.policy import Policy

from smarts.core.agent_interface import RGB, AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register


def entry_point(**kwargs):
    interface = AgentInterface(
        action=ActionSpaceType.Continuous,
        drivable_area_grid_map=False,
        lane_positions=True,
        lidar_point_cloud=False,
        occupancy_grid_map=False,
        road_waypoints=False,
        signals=False,
        top_down_rgb=RGB(
            width = 112,
            height = 112,
            resolution = 50 / 112, # m/pixels
        ),
    )

    agent_params = kwargs["agent_params"]
    agent_params.update({"top_down_rgb": interface.top_down_rgb})

    return AgentSpec(
        interface=interface,
        agent_builder=Policy,
        agent_params=agent_params,
    )


register(locator="contrib-agent-v0", entry_point=entry_point)
