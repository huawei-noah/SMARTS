import unittest
from ultra.baselines.ppo.policy import PPOPolicy
import gym, ray
from ultra.utils.episode import episodes
import numpy as np
from ultra.env.agent_spec import UltraAgentSpec
from smarts.core.controllers import ActionSpaceType

AGENT_ID = "001"
timestep_sec = 0.1
seed = 2
task_id = "00"
task_level = "easy"


class EpisodeTest(unittest.TestCase):
    def test_episode_record(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()
            result = {
                "episode_reward": 0,
                "dist_center": 0,
                "goal_dist": 0,
                "speed": 0,
                "ego_num_violations": 0,
                "linear_jerk": 0,
                "angular_jerk": 0,
                "collision": 0,
                "off_road": 0,
                "off_route": 0,
                "reached_goal": 0,
            }
            for episode in episodes(1, etag="Train"):
                observations = env.reset()
                total_step = 0
                episode.reset()
                dones, infos = {"__all__": False}, None
                state = observations[AGENT_ID]["state"]

                while not dones["__all__"] and total_step < 4:
                    action = agent.act(state, explore=True)
                    observations, rewards, dones, infos = env.step({AGENT_ID: action})
                    next_state = observations[AGENT_ID]["state"]
                    observations[AGENT_ID]["ego"].update(rewards[AGENT_ID]["log"])
                    loss_output = agent.step(
                        state=state,
                        action=action,
                        reward=rewards[AGENT_ID]["reward"],
                        next_state=next_state,
                        done=dones[AGENT_ID],
                        max_steps_reached=observations[AGENT_ID]["ego"][
                            "events"
                        ].reached_max_episode_steps,
                    )

                    for key in result.keys():
                        if key in observations[AGENT_ID]["ego"]:

                            if key == "goal_dist":
                                result[key] = observations[AGENT_ID]["ego"][key]
                            else:
                                result[key] += observations[AGENT_ID]["ego"][key]
                        elif key == "episode_reward":
                            result[key] += rewards[AGENT_ID]["reward"]

                    episode.record_step(
                        agent_id=AGENT_ID,
                        observations=observations,
                        rewards=rewards,
                        total_step=total_step,
                        loss_output=loss_output,
                    )

                    state = next_state
                    total_step += 1
            env.close()
            episode.record_episode()
            return result, episode

        ray.init(ignore_reinit_error=True)
        result, episode = ray.get(run_experiment.remote())
        for key in result.keys():
            self.assertTrue(True)
            if key in ["episode_reward", "goal_dist"]:
                self.assertTrue(
                    abs(result[key] - episode.info["Train"].data[key]) <= 0.001
                )
            else:
                temp = key
                if key == "linear_jerk":
                    temp = "ego_linear_jerk"
                elif key == "angular_jerk":
                    temp = "ego_angular_jerk"
                self.assertTrue(
                    abs(result[key] / episode.steps - episode.info["Train"].data[temp])
                    <= 0.001
                )

    def test_episode_counter(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()
            episode_count = 0
            for episode in episodes(2, etag="Train"):
                observations = env.reset()
                total_step = 0
                episode.reset()
                dones, infos = {"__all__": False}, None
                state = observations[AGENT_ID]["state"]
                while not dones["__all__"]:
                    action = agent.act(state, explore=True)
                    observations, rewards, dones, infos = env.step({AGENT_ID: action})
                    next_state = observations[AGENT_ID]["state"]
                    observations[AGENT_ID]["ego"].update(rewards[AGENT_ID]["log"])
                    loss_output = agent.step(
                        state=state,
                        action=action,
                        reward=rewards[AGENT_ID]["reward"],
                        next_state=next_state,
                        done=dones[AGENT_ID],
                        max_steps_reached=observations[AGENT_ID]["ego"][
                            "events"
                        ].reached_max_episode_steps,
                    )
                    episode.record_step(
                        agent_id=AGENT_ID,
                        observations=observations,
                        rewards=rewards,
                        total_step=total_step,
                        loss_output=loss_output,
                    )
                    state = next_state
                    total_step += 1
                episode_count += 1
            env.close()
            return episode_count

        ray.init(ignore_reinit_error=True)
        episode_count = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue(episode_count == 2)

    def test_tensorboard_handle(self):
        # TODO test numbers
        self.assertTrue(True)

    def test_save_model(self):
        self.assertTrue(True)

    def test_save_code(self):
        self.assertTrue(True)


def prepare_test_env_agent(headless=True):
    timestep_sec = 0.1
    # [throttle, brake, steering]
    policy_class = PPOPolicy
    spec = UltraAgentSpec(
        action_type=ActionSpaceType.Continuous,
        policy_class=policy_class,
        max_episode_steps=10,
    )
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=("00", "easy"),
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )
    agent = spec.build_agent()
    return agent, env
