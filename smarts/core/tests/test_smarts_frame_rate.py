import pytest

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from helpers.scenario import temp_scenario
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.frame_monitor import FrameMonitor
from smarts.core.agent_interface import ActionSpaceType, AgentInterface

AGENT_1 = "Agent_007"


@pytest.fixture
def scenarios():
    with temp_scenario(name="6lane", map="maps/6lane.net.xml") as scenario_root:
        actors = [
            t.SocialAgentActor(
                name=f"non-interactive-agent-{speed}-v0",
                agent_locator="zoo.policies:non-interactive-agent-v0",
                policy_kwargs={"speed": speed},
            )
            for speed in [10, 30, 80]
        ]

        def to_mission(start_edge, end_edge):
            route = t.Route(begin=(start_edge, 1, 0), end=(end_edge, 1, "max"))
            return t.Mission(route=route)

        def fifth_mission(start_edge, end_edge):
            route = t.Route(begin=(start_edge, 0, 0), end=(end_edge, 0, "max"))
            return t.Mission(route=route)

        gen_scenario(
            t.Scenario(
                social_agent_missions={
                    "group-1": (actors, [to_mission("edge-north-NS", "edge-south-NS")]),
                    "group-2": (actors, [to_mission("edge-west-WE", "edge-east-WE")]),
                    "group-3": (actors, [to_mission("edge-east-EW", "edge-west-EW")]),
                    "group-4": (actors, [to_mission("edge-south-SN", "edge-north-SN")]),
                    "group-5": (
                        actors,
                        [fifth_mission("edge-south-SN", "edge-east-WE")],
                    ),
                },
                ego_missions=[
                    t.Mission(
                        t.Route(
                            begin=("edge-west-WE", 0, 0), end=("edge-east-WE", 0, "max")
                        )
                    )
                ],
            ),
            output_dir=scenario_root,
        )
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_1]
        )


@pytest.fixture
def smarts():
    laner = AgentInterface(
        max_episode_steps=1000,
        action=ActionSpaceType.Lane,
    )

    agents = {AGENT_1: laner}
    smarts = SMARTS(
        agents,
        traffic_sim=SumoTrafficSimulation(headless=True),
        envision=None,
    )

    yield smarts
    smarts.destroy()


def test_smarts_framerate(smarts, scenarios):
    scenario = next(scenarios)
    smarts.reset(scenario)

    for _ in range(10):
        with FrameMonitor(30):
            smarts.step({AGENT_1: "keep_lane"})
