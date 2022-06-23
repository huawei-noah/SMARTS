import os
import os.path
import pickle
import numpy as np
import random
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL, BCQ
import pathlib
import glob

data_size = len(os.listdir('collected_data'))
index = list(range(data_size))
index = [i + 1 for i in index]
total_num_train = 0
for i in index:

    obs = list()
    actions = list()
    rewards = list()
    terminals = list()
    filename = str(i) + '-EGO.pkl'
    print('Training with ' + filename)
    if os.path.isfile('saved_model/model_' + str(i) + '.pt'):
        print('skip ' + filename)
        total_num_train += 1
        continue
    else:
        try:       

            with open('collected_data/' +  filename, 'rb') as f:
                data = pickle.load(f)
            print('There are ' + str(len(data)) + ' slices')


            for j in range(len(data)):
                if data[j]['rewards'] != None:
                    obs.append(data[j]['obs'].top_down_rgb.data.reshape(3,256,256))
                    actions.append(data[j]['actions'])
                    rewards.append(data[j]['rewards'])
                    if data[j]['dones']:
                        terminals.append(1)
                    else:
                        terminals.append(0)
            obs = np.array(obs)
            actions = np.array(actions)
            rewards = np.array(rewards)
            terminals = np.array(terminals)
            dataset = MDPDataset(obs, actions, rewards, terminals)

            if total_num_train == 0:
                model = d3rlpy.algos.CQL(use_gpu=True, batch_size=1)
                model.fit(dataset, 
                    eval_episodes=dataset, 
                    n_steps_per_epoch = 100,
                    n_steps = 100, 
                )
                model.save_model('saved_model/model_' + str(i) + '.pt')
            else:
                saved_models = glob.glob('saved_model/*')
                latest_model = max(saved_models, key=os.path.getctime)
                saved_folder = sorted(os.listdir(pathlib.Path(__file__).absolute().parent/'d3rlpy_logs/'))[0]
                model = CQL.from_json('d3rlpy_logs/' + saved_folder + '/params.json', use_gpu=True)
                model.load_model(latest_model)
                print('Load model_' + latest_model)
                model.fit(dataset, 
                    eval_episodes=dataset, 
                    n_steps_per_epoch = 100,
                    n_steps = 100, 
                    save_metrics=False
                )
                model.save_model('saved_model/model_' + str(i) + '.pt')
            
            total_num_train += 1
            print('There are ' + str(total_num_train) + ' being trained')
        except:
            total_num_train += 1
       
 
        



