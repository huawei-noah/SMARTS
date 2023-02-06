from pathlib import Path
from typing import Any, Dict

from smarts.core.agent import Agent
from zoo.utils.command import run as runcmd

from .observation import observation_adapter
from .planner import *
from .predictor import *


class Policy(Agent):
    def __init__(self):
        model = Path(__file__).absolute().parents[0] / "predictor_5000_0.6726.pth"
        if not model.exists():
            # Download trained model.
            runcmd(
                "wget -O model https://github.com/smarts-project/smarts-project.rl/blob/master/interaction_aware_motion_prediction/predictor_5000_0.6726.pth"
            )

        self.predictor = Predictor()
        self.predictor.load_state_dict(torch.load(model, map_location="cpu"))
        self.predictor.eval()
        self.planner = Planner(self.predictor)
        self.observer = observation_adapter(num_neighbors=5)
        self.cycle = 3
        self.actions = []

    def act(self, obs: Dict[str, Any]):

        # Reset
        if obs.steps_completed == 1:
            self.observer.reset()
            self.actions = []

        # Pre-process
        wrapped_ob = self.observer(obs)
        wrapped_obs = (obs, wrapped_ob)
        self.observer.timestep += 1

        # Use previously planned trajectory
        if len(self.actions) >= 1:
            return self.actions.pop(0)

        # Act
        self.actions = self.planner.plan(wrapped_obs)
        self.actions = self.actions[: self.cycle]

        return self.actions.pop(0)
