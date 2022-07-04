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
import glob
import pandas as pd

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


# saved_models = glob.glob('d3rlpy_logs/*')
# models = sorted(saved_models, key=os.path.getctime)
# values = list()
# # df = pd.DataFrame()
# cols = ['values', 'td', 'acc_rewards']
# for dir in models:
#     value = pd.read_csv(dir + '/value_scale.csv')
#     td = pd.read_csv(dir + '/td_error.csv')
#     model = CQL.from_json(dir + '/params.json', use_gpu=True)
#     model.load_model(dir + '/model_100.pt')
#     obs=env.reset()
#     acc_reward = 0
#     # print(obs)
#     done = False
#     while not done:
#         bev = np.array([obs['EGO'].top_down_rgb.data.reshape(3,256,256)])
#         action = model.predict(bev)[0]
#         agent_actions = {'EGO': action}
#         obs, reward, done, extra = env.step(agent_actions)
#         acc_reward += reward['EGO']
#         done = done['EGO']
#     values.append([float(value.columns[-1]), float(td.columns[-1]), acc_reward])
# df = pd.DataFrame(np.array(values), columns=cols)
# df.to_csv('values.csv', index=False)

# env.close()



# evaluate using offline data




with open('collected_data/2000-EGO.pkl', 'rb') as f:
    data = pickle.load(f)
    print(len(data))

saved_models = glob.glob('d3rlpy_logs/*')
models = sorted(saved_models, key=os.path.getctime)
offline_rewards = list()
for dir in models:
    acc_reward = 0
    value = pd.read_csv(dir + '/value_scale.csv')
    td = pd.read_csv(dir + '/td_error.csv')
    model = CQL.from_json(dir + '/params.json', use_gpu=True)
    model.load_model(dir + '/model_100.pt')
    # model.load_model('d3rlpy_logs/100/model_100.pt')
    ob = env.reset()
    done = False
    for j in range(len(data)):
        if not done:
            ob = data[j]['obs']
            bev = np.array([data[j]['obs'].top_down_rgb.data.reshape(3,256,256)])
            action = model.predict(bev)[0]
            agent_actions = {'EGO': action}

            ob, reward, done, extra = env.step(agent_actions)

            acc_reward += reward['EGO']
            done = done['EGO']
        else:
            continue
    offline_rewards.append(acc_reward)
df = pd.read_csv('values.csv')
df['offline_rewards'] = offline_rewards
df.to_csv('values.csv', index=False)




