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

def _build_scenario():
    scenario = str(pathlib.Path(__file__).absolute().parent / "scenarios/cruise")
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)

max_episode_steps = 300
env = CompetitionEnv(["scenarios/cruise"], max_episode_steps=300, headless=False)
_build_scenario()
print("Start evaluation.")
saved_folder = sorted(os.listdir(pathlib.Path(__file__).absolute().parent/'d3rlpy_logs/'))[0]
model = CQL.from_json('d3rlpy_logs/' + saved_folder + '/params.json', use_gpu=True)
model.load_model('saved_model/model_500.pt')
for i in range(10):
    obs=env.reset()
    # print(obs)
    done = False
    while not done:
        bev = np.array([obs['EGO'].top_down_rgb.data.reshape(3,256,256)])
        action = model.predict(bev)[0]
        agent_actions = {'EGO': action}
        print(action)
        obs, reward, done, extra = env.step(agent_actions)
        done = done['EGO']
env.close()

# with open('collected_data/100-EGO.pkl', 'rb') as f:
#     data = pickle.load(f)

# obs = list()
# for j in range(len(data)):
#     if data[j]['rewards'] != None:
#         obs.append(data[j]['obs'].top_down_rgb.data.reshape(3,256,256))
# obs = np.array(obs)
# actions = model.predict(obs)
# print(actions)
# print(actions.shape)
