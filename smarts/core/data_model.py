from dataclasses import dataclass, field
from typing import Dict, Any

from smarts.core.agent import AgentSpec
from smarts.sstudio.types import Mission
from smarts.zoo.registry import make


@dataclass
class SocialAgent:
    id: str
    name: str
    agent_locator: str
    mission: Mission
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    initial_speed: float = None

    def to_agent_spec(self) -> AgentSpec:
        return make(locator=self.agent_locator, **self.policy_kwargs)
