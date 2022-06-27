import os
import os.path
import pickle
import numpy as np
import random
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL
import pathlib
from competition_env import CompetitionEnv

from d3rlpy.ope import FQE
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer

def _build_scenario():
    scenario = str(pathlib.Path(__file__).absolute().parent / "scenarios/cruise")
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)

max_episode_steps = 300
env = CompetitionEnv(["scenarios/cruise"], max_episode_steps=300, headless=True)
_build_scenario()
print("Start evaluation.")
files = os.listdir('saved_model')
files.sort(key=os.path.getctime)
for filename in files:
    saved_folder = sorted(os.listdir(pathlib.Path(__file__).absolute().parent/'d3rlpy_logs/'))[0]
    model = CQL.from_json('d3rlpy_logs/' + saved_folder + '/params.json', use_gpu=True)
    model.load_model('saved_model/' + filename)
    print('Loading model ' + filename)

    # evaluate using simulation
    eval_rewards = list()
    for i in range(1):
        obs=env.reset()
        acc_reward = 0
        # print(obs)
        done = False
        while not done:
            bev = np.array([obs['EGO'].top_down_rgb.data.reshape(3,256,256)])
            action = model.predict(bev)[0]
            agent_actions = {'EGO': action}
            obs, reward, done, extra = env.step(agent_actions)
            acc_reward += reward['EGO']
            done = done['EGO']
        eval_rewards.append(acc_reward)
    print(eval_rewards)
env.close()


# evaluate using offline data




# with open('collected_data/100-EGO.pkl', 'rb') as f:
#     data = pickle.load(f)

# obs = list()
# actions = list()
# rewards = list()
# terminals = list()
# for j in range(len(data)):
#     if data[j]['rewards'] != None:
#         obs.append(data[j]['obs'].top_down_rgb.data.reshape(3,256,256))
#         actions.append(data[j]['actions'])
#         rewards.append(data[j]['rewards'])
#         if data[j]['dones']:
#             terminals.append(1)
#         else:
#             terminals.append(0)
# obs = np.array(obs)
# actions = np.array(actions)
# rewards = np.array(rewards)
# terminals = np.array(terminals)
# dataset = MDPDataset(obs, actions, rewards, terminals)
# fqe = FQE(algo=model)
# fqe.fit(dataset.episodes,
#         eval_episodes=dataset.episodes,
#         n_steps=10,
#         n_steps_per_epoch = 10,
#         scorers={
#            'init_value': initial_state_value_estimation_scorer,
#            'soft_opc': soft_opc_scorer(return_threshold=180)
#         })

