from ray import tune

config = {
    "action_size": 2,
    "batch_size": 2048,  # size of batch
    "lr": 3e-5,
    "mini_batch_size": 64,  # 64
    "epoch_count": 20,
    "gamma": 0.99,  # discounting
    "l": 0.95,  # lambda used in lambda-return
    "eps": 0.2,  # epsilon value used in PPO clipping
    "critic_tau": 1.0,
    "actor_tau": 1.0,
    "entropy_tau": 0.0,
    "hidden_units": 512,
    "seed": 2,
    "logging_freq": 2,
    "observation_num_lookahead": 20,
    "social_vehicles":{
        "encoder_key": tune.choice(['no_encoder', 'precog_encoder', 'pointnet_encoder']),
        "social_policy_hidden_units": 128,
        "social_policy_init_std": 0.5,
        "social_capacity": 10,
        "num_social_features": 4,
        "seed": 2
    }
}
