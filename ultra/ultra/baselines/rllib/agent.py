from ultra.baselines.rllib.models.fc_network import CustomFCModel


class RllibAgent:
    def __init__(self, agent_type):
        self.agent_type = agent_type

    def train(self, env, logger_creator, config):
        if self.agent_type=='ppo':
            return ppo.PPOTrainer(
                env=env,
                config=config,
                logger_creator=logger_creator,
            ).train()
        elif self.agent_type=='ddpg':
            return  ddpg.DDPGTrainer(
                env=env,
                config=config,
                logger_creator=logger_creator,
            ).train()
        elif self.agent_type=='td3':
            return td3.TD3Trainer(
                env=env,
                config=config,
                logger_creator=logger_creator,
            ).train()
        # elif self.agent_type=='dqn':
        #     trainer = dqn.DQNTrainer(
        #         env=RLlibUltraEnv,
        #         config=tune_config,
        #         logger_creator=log_creator(log_dir),
        #     )
