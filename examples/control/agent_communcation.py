import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1
from smarts.env.gymnasium.wrappers.agent_communication import (
    Bands,
    Header,
    Message,
    MessagePasser,
    V2XReceiver,
    V2XTransmitter,
)
from smarts.env.utils.action_conversion import ActionOptions
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.sstudio.scenario_construction import build_scenarios

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))
import gymnasium as gym

from examples.tools.argument_parser import default_argument_parser

TIMESTEP = 0.1
BYTES_IN_MEGABIT = 125000
MESSAGE_MEGABITS_PER_SECOND = 10
MESSAGE_BYTES = int(BYTES_IN_MEGABIT * MESSAGE_MEGABITS_PER_SECOND / TIMESTEP)


def filter_useless(
    transmissions: List[Header, Message]
) -> Generator[Tuple[Header, Message], None, None]:
    """A primitive example filter that takes in transmissions and outputs filtered transmissions."""
    for header, msg in transmissions:
        if header.sender in ("parked_agent", "broken_stoplight"):
            continue
        if header.sender_type in ("advertisement",):
            continue
        yield header, msg


class LaneFollowerAgent(Agent):
    def act(self, obs: Dict[Any, Union[Any, Dict]]):
        return (obs["waypoint_paths"]["speed_limit"][0][0], 0)


class GossiperAgent(Agent):
    def __init__(self, id_: str, base_agent: Agent, filter_, friends):
        self._filter = filter_
        self._id = id_
        self._friends = friends
        self._base_agent = base_agent

    def act(self, obs, **configs):
        out_transmissions = []
        for header, msg in self._filter(obs["transmissions"]):
            header: Header = header
            msg: Message = msg
            if not {self._id, "__all__"}.intersection(header.cc | header.bcc):
                continue
            if header.channel == "position_request":
                print()
                print("On step: ", obs["steps_completed"])
                print("Gossiper received position request: ", header)
                out_transmissions.append(
                    (
                        Header(
                            channel="position",
                            sender=self._id,
                            sender_type="ad_vehicle",
                            cc={header.sender},
                            bcc={*self._friends},
                            format="position",
                        ),  # optimize this later
                        Message(
                            content=obs["ego_vehicle_state"]["position"],
                        ),  # optimize this later
                    )
                )
                print("Gossiper sent position: ", out_transmissions[0][1])

        base_action = self._base_agent.act(obs)
        return (base_action, out_transmissions)


class SchemerAgent(Agent):
    def __init__(self, id_: str, base_agent: Agent, request_freq) -> None:
        self._base_agent = base_agent
        self._id = id_
        self._request_freq = request_freq

    def act(self, obs, **configs):
        out_transmissions = []
        for header, msg in obs["transmissions"]:
            header: Header = header
            msg: Message = msg
            if header.channel == "position":
                print()
                print("On step: ", obs["steps_completed"])
                print("Schemer received position: ", msg)

        if obs["steps_completed"] % self._request_freq == 0:
            print()
            print("On step: ", obs["steps_completed"])
            out_transmissions.append(
                (
                    Header(
                        channel="position_request",
                        sender=self._id,
                        sender_type="ad_vehicle",
                        cc=set(),
                        bcc={"__all__"},
                        format="position_request",
                    ),
                    Message(content=None),
                )
            )
            print("Schemer requested position with: ", out_transmissions[0][0])

        base_action = self._base_agent.act(obs)
        return (base_action, out_transmissions)


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_interface = AgentInterface.from_type(
        AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
    )
    hiwayv1env = HiWayEnvV1(
        scenarios=scenarios,
        agent_interfaces={"gossiper0": agent_interface, "schemer": agent_interface},
        headless=headless,
        observation_options=ObservationOptions.multi_agent,
        action_options=ActionOptions.default,
    )
    # for now
    env = MessagePasser(
        hiwayv1env,
        max_message_bytes=MESSAGE_BYTES,
        message_config={
            "gossiper0": (
                V2XTransmitter(
                    bands=Bands.ALL,
                    range=100,
                    # available_channels=["position_request", "position"]
                ),
                V2XReceiver(
                    bands=Bands.ALL,
                    aliases=["tim"],
                    blacklist_channels={"self_control"},
                ),
            ),
            "schemer": (
                V2XTransmitter(
                    bands=Bands.ALL,
                    range=100,
                ),
                V2XReceiver(
                    bands=Bands.ALL,
                    aliases=[],
                ),
            ),
        },
    )
    agents = {
        "gossiper0": GossiperAgent(
            "gossiper0",
            base_agent=LaneFollowerAgent(),
            filter_=filter_useless,
            friends={"schemer"},
        ),
        "schemer": SchemerAgent(
            "schemer", base_agent=LaneFollowerAgent(), request_freq=100
        ),
    }

    # then just the standard gym interface with no modifications
    for episode in episodes(n=num_episodes):
        observation, info = env.reset()
        episode.record_scenario(env.scenario_log)

        terminated = {"__all__": False}
        while not terminated["__all__"]:
            agent_action = {
                agent_id: agents[agent_id].act(obs)
                for agent_id, obs in observation.items()
            }
            observation, reward, terminated, truncated, info = env.step(agent_action)
            episode.record_step(observation, reward, terminated, info)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(Path(__file__).absolute().parents[2] / "scenarios" / "sumo" / "loop")
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
