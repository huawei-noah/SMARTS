import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.sac as sac
import ray.rllib.agents.ddpg as ddpg


class RllibAgent:
    def __init__(self, agent_type, env, config, logger_creator):
        assert agent_type in ["td3", "ddpg", "dqn", "ppo", "sac"]
        self.agent_type = agent_type

        if self.agent_type == "ppo":
            self.trainer = ppo.PPOTrainer(
                env=env,
                config=config,
                logger_creator=logger_creator,
            )
        elif self.agent_type == "ddpg":
            self.trainer = ddpg.DDPGTrainer(
                env=env,
                config=config,
                logger_creator=logger_creator,
            )
        elif self.agent_type == "td3":
            self.trainer = td3.TD3Trainer(
                env=env,
                config=config,
                logger_creator=logger_creator,
            )
        # elif self.agent_type=='dqn':
        #     trainer = dqn.DQNTrainer(
        #         env=RLlibUltraEnv,
        #         config=tune_config,
        #         logger_creator=log_creator(log_dir),
        #     )

    def train(self):
        return self.trainer.train()

    @staticmethod
    def rllib_default_config(agent_type):
        assert agent_type in ["td3", "ddpg", "dqn", "ppo", "sac"]

        if agent_type == "ppo":
            return ppo.DEFAULT_CONFIG.copy()
        elif agent_type == "ddpg":
            return ddpg.DEFAULT_CONFIG.copy()
        elif agent_type == "td3":
            return td3.DEFAULT_CONFIG.copy()
