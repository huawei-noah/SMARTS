from ultra.baselines.rllib.models.fc_network import CustomFCModel


class RllibAgent:
    def __init__(self, agent_type):
        self.agent_type = agent_type
        if self.agent_type == 'td3':
            policy_params = load_yaml(f"ultra/baselines/ddpg/ddpg/params.yaml")
        else:
            policy_params = load_yaml(f"ultra/baselines/{self.agent_type}/{self.agent_type}/params.yaml")

    def train(self):
        if self.agent_type=='ppo':
            return ppo.PPOTrainer
            # (
            #     env=RLlibUltraEnv,
            #     config=tune_config,
            #     logger_creator=log_creator(log_dir),
            # )
        elif self.agent_type=='ddpg':
            return  ddpg.DDPGTrainer
            # (
            #     env=RLlibUltraEnv,
            #     config=tune_config,
            #     logger_creator=log_creator(log_dir),
            # )
        elif self.agent_type=='td3':
            return td3.TD3Trainer
            # (
            #     env=RLlibUltraEnv,
            #     config=tune_config,
            #     logger_creator=log_creator(log_dir),
            # )
        # elif self.agent_type=='dqn':
        #     trainer = dqn.DQNTrainer(
        #         env=RLlibUltraEnv,
        #         config=tune_config,
        #         logger_creator=log_creator(log_dir),
        #     )
