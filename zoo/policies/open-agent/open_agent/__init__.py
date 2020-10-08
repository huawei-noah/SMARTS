from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register

from .policy import Policy


def entrypoint(
    N=15,
    SV_N=7,
    WP_N=20,
    ts=1.0,
    Q_theta=0,
    Q_position=12,
    Q_obstacle=1500,
    Q_u_accel=30,
    Q_u_yaw_rate=1,
    Q_n=30,
    Q_impatience=0.1,
    debug=False,
    retries=5,
):
    return AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.Trajectory,
            waypoints=True,
            neighborhood_vehicles=True,
            max_episode_steps=None,
        ),
        policy_params={
            "N": N,
            "SV_N": SV_N,
            "WP_N": WP_N,
            "ts": ts,
            "Q_theta": Q_theta,
            "Q_position": Q_position,
            "Q_obstacle": Q_obstacle,
            "Q_u_accel": Q_u_accel,
            "Q_u_yaw_rate": Q_u_yaw_rate,
            "Q_n": Q_n,
            "Q_impatience": Q_impatience,
            "debug": debug,
            "retries": retries,
        },
        policy_builder=Policy,
        perform_self_test=False,
    )


register(locator="open_agent-v0", entry_point=entrypoint)
