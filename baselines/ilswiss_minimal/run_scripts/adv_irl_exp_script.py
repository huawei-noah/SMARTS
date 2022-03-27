import yaml
import argparse
import numpy as np
import os
import sys
import inspect
import random
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from rlkit.envs import get_env, get_envs

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.algorithms.sac.sac_alpha import SoftActorCritic
from rlkit.torch.algorithms.adv_irl.disc_models.simple_disc_models import MLPDisc
from rlkit.torch.algorithms.adv_irl.adv_irl import AdvIRL
from rlkit.envs.wrappers import ProxyEnv, NormalizedBoxActEnv, ObsScaledEnv, EPS


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)

    demos_path = listings[variant["expert_name"]]["file_paths"][variant["expert_idx"]]
    """
    Buffer input format
    """
    # buffer_save_dict = joblib.load(expert_demos_path)
    # expert_replay_buffer = buffer_save_dict['train']
    # obs_mean, obs_std = buffer_save_dict['obs_mean'], buffer_save_dict['obs_std']
    # acts_mean, acts_std = buffer_save_dict['acts_mean'], buffer_save_dict['acts_std']
    # obs_min, obs_max = buffer_save_dict['obs_min'], buffer_save_dict['obs_max']
    # if 'minmax_env_with_demo_stats' in variant.keys():
    #     if (variant['minmax_env_with_demo_stats']) and not (variant['scale_env_with_demo_stats']):
    #         assert 'norm_train' in buffer_save_dict.keys()
    #         expert_replay_buffer = buffer_save_dict['norm_train']
    """
    PKL input format
    """
    print("demos_path", demos_path)
    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)
    if variant["traj_num"] > 0:
        traj_list = random.sample(traj_list, variant["traj_num"])

    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_creator"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space_n))
    print("Act Space: {}\n\n".format(env.action_space_n))

    expert_replay_buffer = EnvReplayBuffer(
        variant["adv_irl_params"].get(
            "expert_buffer_size", variant["adv_irl_params"]["replay_buffer_size"]
        ),
        env,
        random_seed=np.random.randint(10000),
    )

    if "expert_buffer_size" in variant["adv_irl_params"]:
        variant["adv_irl_params"].pop("expert_buffer_size")

    obs_space_n = env.observation_space_n
    act_space_n = env.action_space_n

    policy_mapping_dict = dict(
        zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
    )

    policy_trainer_n = {}
    policy_n = {}
    disc_model_n = {}

    for agent_id in env.agent_ids:
        policy_id = policy_mapping_dict.get(agent_id)
        if policy_id not in policy_trainer_n:
            print(f"Create {policy_id} for {agent_id} ...")
            obs_space = obs_space_n[agent_id]
            act_space = act_space_n[agent_id]
            assert isinstance(obs_space, gym.spaces.Box)
            assert isinstance(act_space, gym.spaces.Box)
            assert len(obs_space.shape) == 1
            assert len(act_space.shape) == 1

            obs_dim = obs_space.shape[0]
            action_dim = act_space.shape[0]

            # build the policy models
            net_size = variant["policy_net_size"]
            num_hidden = variant["policy_num_hidden_layers"]
            qf1 = FlattenMlp(
                hidden_sizes=num_hidden * [net_size],
                input_size=obs_dim + action_dim,
                output_size=1,
            )
            qf2 = FlattenMlp(
                hidden_sizes=num_hidden * [net_size],
                input_size=obs_dim + action_dim,
                output_size=1,
            )
            vf = FlattenMlp(
                hidden_sizes=num_hidden * [net_size],
                input_size=obs_dim,
                output_size=1,
            )
            policy = ReparamTanhMultivariateGaussianPolicy(
                hidden_sizes=num_hidden * [net_size],
                obs_dim=obs_dim,
                action_dim=action_dim,
            )

            # build the discriminator model
            disc_model = MLPDisc(
                obs_dim + action_dim
                if not variant["adv_irl_params"]["state_only"]
                else 2 * obs_dim,
                num_layer_blocks=variant["disc_num_blocks"],
                hid_dim=variant["disc_hid_dim"],
                hid_act=variant["disc_hid_act"],
                use_bn=variant["disc_use_bn"],
                clamp_magnitude=variant["disc_clamp_magnitude"],
            )

            # set up the algorithm
            trainer = SoftActorCritic(
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                vf=vf,
                action_space=env.action_space_n[agent_id],
                **variant["sac_params"],
            )

            policy_trainer_n[policy_id] = trainer
            policy_n[policy_id] = policy
            disc_model_n[policy_id] = disc_model
        else:
            print(f"Use existing {policy_id} for {agent_id} ...")

    env_wrapper = ProxyEnv  # Identical wrapper
    for act_space in act_space_n.values():
        if isinstance(act_space, gym.spaces.Box):
            env_wrapper = NormalizedBoxActEnv
            break

    if variant["scale_env_with_demo_stats"]:
        obs = np.vstack(
            [
                traj_list[i][k]["observations"]
                for i in range(len(traj_list))
                for k in traj_list[i].keys()
            ]
        )
        obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)
        print("mean:{}\nstd:{}".format(obs_mean, obs_std))

        _env_wrapper = env_wrapper
        env_wrapper = lambda *args, **kwargs: ObsScaledEnv(
            _env_wrapper(*args, **kwargs),
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        for i in range(len(traj_list)):
            for k in traj_list[i].keys():
                traj_list[i][k]["observations"] = (
                    traj_list[i][k]["observations"] - obs_mean
                ) / (obs_std + EPS)
                traj_list[i][k]["next_observations"] = (
                    traj_list[i][k]["next_observations"] - obs_mean
                ) / (obs_std + EPS)

    env = env_wrapper(env)

    for i in range(len(traj_list)):
        expert_replay_buffer.add_path(traj_list[i])

    print(
        "Load {} trajectories, {} samples".format(
            len(traj_list), expert_replay_buffer.num_steps_can_sample()
        )
    )

    train_split_path = listings[variant["expert_name"]]["train_split"][0]
    with open(train_split_path, "rb") as f:
        train_vehicle_ids = pickle.load(f)
    train_vehicle_ids_list = np.array_split(
        train_vehicle_ids,
        env_specs["training_env_specs"]["env_num"],
    )

    print(
        "Creating {} training environments, each with {} vehicles ...".format(
            env_specs["training_env_specs"]["env_num"], len(train_vehicle_ids_list[0])
        )
    )
    training_env = get_envs(
        env_specs,
        env_wrapper,
        vehicle_ids_list=train_vehicle_ids_list,
        **env_specs["training_env_specs"],
    )

    eval_split_path = listings[variant["expert_name"]]["eval_split"][0]
    with open(eval_split_path, "rb") as f:
        eval_vehicle_ids = pickle.load(f)
    eval_vehicle_ids_list = np.array_split(
        eval_vehicle_ids,
        env_specs["eval_env_specs"]["env_num"],
    )

    print(
        "Creating {} evaluation environments, each with {} vehicles ...".format(
            env_specs["eval_env_specs"]["env_num"], len(eval_vehicle_ids_list[0])
        )
    )
    eval_env = get_envs(
        env_specs,
        env_wrapper,
        vehicle_ids_list=eval_vehicle_ids_list,
        **env_specs["eval_env_specs"],
    )
    eval_car_num = np.array([len(v_ids) for v_ids in eval_vehicle_ids_list])

    algorithm = AdvIRL(
        env=env,
        training_env=training_env,
        eval_env=eval_env,
        exploration_policy_n=policy_n,
        policy_mapping_dict=policy_mapping_dict,
        discriminator_n=disc_model_n,
        policy_trainer_n=policy_trainer_n,
        expert_replay_buffer=expert_replay_buffer,
        eval_car_num=eval_car_num,
        **variant["adv_irl_params"],
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.FullLoader)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    exp_suffix = ""
    exp_suffix = "--gp-{}--rs-{}--trajnum-{}".format(
        exp_specs["adv_irl_params"]["grad_pen_weight"],
        exp_specs["sac_params"]["reward_scale"],
        format(exp_specs["traj_num"]),
    )

    if "obs_stacked_size" in exp_specs["env_specs"]["env_kwargs"]:
        exp_suffix += "--stack-{}".format(
            exp_specs["env_specs"]["env_kwargs"]["obs_stacked_size"]
        )

    if not exp_specs["adv_irl_params"]["no_terminal"]:
        exp_suffix = "--terminal" + exp_suffix

    if exp_specs["scale_env_with_demo_stats"]:
        exp_suffix = "--scale" + exp_suffix

    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)

    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)

    exp_prefix = exp_prefix + exp_suffix
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    experiment(exp_specs)
