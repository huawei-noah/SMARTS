from smarts.zoo.registry import register
from .sac.sac.policy import SACPolicy
from .ppo.ppo.policy import PPOPolicy
from .dqn.dqn.policy import DQNPolicy
from .ddpg.ddpg.policy import TD3Policy
from .bdqn.bdqn.policy import BehavioralDQNPolicy
from smarts.core.controllers import ActionSpaceType
from ultra.baselines.agent_spec import UltraAgentSpec

register(
    locator="sac-v0",
    entry_point=lambda **kwargs: UltraAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=SACPolicy, **kwargs
    ),
)
register(
    locator="ppo-v0",
    entry_point=lambda **kwargs: UltraAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=PPOPolicy, **kwargs
    ),
)
register(
    locator="ddpg-v0",
    entry_point=lambda **kwargs: UltraAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=TD3Policy, **kwargs
    ),
)
register(
    locator="dqn-v0",
    entry_point=lambda **kwargs: UltraAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=DQNPolicy, **kwargs
    ),
)
register(
    locator="bdqn-v0",
    entry_point=lambda **kwargs: UltraAgentSpec(
        action_type=ActionSpaceType.Lane, policy_class=BehavioralDQNPolicy, **kwargs
    ),
)
