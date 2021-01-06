# import unittest, ray, glob, gym, os
# from ultra.baselines.ppo.policy import PPOPolicy
# from ultra.src.evaluate import evaluate, evaluation_check
# from ultra.utils.episode import episodes
# from ultra.env.agent_spec import UltraAgentSpec
# from smarts.core.controllers import ActionSpaceType
#
# seed = 2
# policy_class = PPOPolicy
# AGENT_ID = "001"
#
#
# class EvaluateTest(unittest.TestCase):
#     def test_evaluation_check(self):
#         # @ray.remote(max_calls=1, num_gpus=0)
#         def run_experiment():
#             total_step = 0
#             agent, env = prepare_test_env_agent(headless=False)
#             timestep_sec1 = env.timestep_sec
#             for episode in episodes(1, etag="Train"):
#                 observations = env.reset()
#                 state = observations[AGENT_ID]["state"]
#                 dones, infos = {"__all__": False}, None
#                 episode.reset()
#                 while not dones["__all__"]:
#                     evaluation_check(
#                         agent=agent,
#                         agent_id=AGENT_ID,
#                         episode=episode,
#                         eval_rate=10,
#                         eval_episodes=1,
#                         policy_class=policy_class,
#                         scenario_info=("00", "easy"),
#                         timestep_sec=0.1,
#                         headless=True,
#                     )
#                     action = agent.act(state, explore=True)
#                     observations, rewards, dones, infos = env.step({AGENT_ID: action})
#                     next_state = observations[AGENT_ID]["state"]
#
#                     # retrieve some relavant information from reward processor
#                     observations[AGENT_ID]["ego"].update(rewards[AGENT_ID]["log"])
#                     loss_output = agent.step(
#                         state=state,
#                         action=action,
#                         reward=rewards[AGENT_ID]["reward"],
#                         next_state=next_state,
#                         done=dones[AGENT_ID],
#                         max_steps_reached=observations[AGENT_ID]["ego"][
#                             "events"
#                         ].reached_max_episode_steps,
#                     )
#                     episode.record_step(
#                         agent_id=AGENT_ID,
#                         observations=observations,
#                         rewards=rewards,
#                         total_step=total_step,
#                         loss_output=loss_output,
#                     )
#                     total_step += 1
#                     state = next_state
#
#             env.close()
#
#         # ray.init(ignore_reinit_error=True)
#         try:
#             run_experiment()
#             self.assertTrue(True)
#         except Exception as err:
#             print(err)
#             self.assertTrue(False)
#
#     def test_evaluate_cli(self):
#         try:
#             os.system(
#                 "python ultra/src/evaluate.py --task 00 --level easy --polic ULTRA_PPO --models ultra/tests/ppo_models/models --episodes 1"
#             )
#             self.assertTrue(True)
#         except Exception as err:
#             print(err)
#             self.assertTrue(False)
#
#     def test_evaluate_module(self):
#         seed = 2
#         policy_class = PPOPolicy
#         # ray.init(ignore_reinit_error=True)
#         model = glob.glob("ultra/tests/ppo_models/models/*")[0]
#         try:
#             evaluate(
#                 agent_id="AGENT_001",
#                 policy_class=policy_class,
#                 seed=seed,
#                 itr_count=0,
#                 checkpoint_dir=model,
#                 scenario_info=("00", "easy"),
#                 num_episodes=2,
#                 timestep_sec=0.1,
#                 headless=True,
#             )
#             self.assertTrue(True)
#         except Exception as err:
#             print(err)
#             self.assertTrue(False)
#
#
# def prepare_test_env_agent(headless=True):
#     timestep_sec = 0.1
#     # [throttle, brake, steering]
#     policy_class = PPOPolicy
#     spec = UltraAgentSpec(
#         action_type=ActionSpaceType.Continuous,
#         policy_class=policy_class,
#         max_episode_steps=10,
#     )
#     env = gym.make(
#         "ultra.env:ultra-v0",
#         agent_specs={AGENT_ID: spec},
#         scenario_info=("00", "easy"),
#         headless=headless,
#         timestep_sec=timestep_sec,
#         seed=seed,
#     )
#     agent = spec.build_agent()
#     return agent, env
