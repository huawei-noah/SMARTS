# import importlib.resources as pkg_resources
# from .policy import SACPolicy
# from ultra.baselines.agent_spec import UltraAgentSpec
#
# VERSION = "0.1.1"
#
#
# def entrypoint(
#     checkpoint_dir=None,
#     task=None,
#     max_episode_steps=1200,
#     experiment_dir=None,
# ):
#     with pkg_resources.path(checkpoint_dir, "checkpoint") as checkpoint_path:
#         return UltraAgentSpec(action_type=ActionSpaceType.Continuous, policy_class=SACPolicy)
#
# # if __name__=='__main__':
# #     entrypoint()
