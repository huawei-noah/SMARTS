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
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from sklearn.model_selection import cross_validate


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
    if os.path.isfile('saved_model_eval/model_' + str(i) + '.pt'):
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
                model = d3rlpy.algos.CQL(use_gpu=True, batch_size=32)
            else:
                saved_models = glob.glob('d3rlpy_logs/*')
                latest_model = max(saved_models, key=os.path.getctime)
                model = CQL.from_json('d3rlpy_logs/1/params.json', use_gpu=True)
                model.load_model(latest_model + '/model_100.pt')
                # saved_models = glob.glob('d3rlpy_logs/*')
                # latest_model = max(saved_models, key=os.path.getctime)
                # saved_folder = sorted(os.listdir(pathlib.Path(__file__).absolute().parent/'d3rlpy_logs/'))[0]

            model.fit(dataset, 
                eval_episodes=dataset, 
                n_steps_per_epoch = 100,
                n_steps = 100, 
                scorers={
                            'td_error': td_error_scorer,
                            'value_scale': average_value_estimation_scorer,
                        }
            )
            
            saved_models = glob.glob('d3rlpy_logs/*')
            latest_model = max(saved_models, key=os.path.getctime)
            os.rename(latest_model, 'd3rlpy_logs/' + str(total_num_train + 1))

            # else:
            #     saved_models = glob.glob('d3rlpy_logs/*')
            #     latest_model = max(saved_models, key=os.path.getctime)
            #     saved_folder = sorted(os.listdir(pathlib.Path(__file__).absolute().parent/'d3rlpy_logs/'))[0]
            #     # model = CQL.from_json('d3rlpy_logs/' + saved_folder + '/params.json', use_gpu=True)
            #     model = CQL.from_json(latest_model + '/params.json', use_gpu=True)
            #     model.load_model(latest_model + '/model_' + str(total_num_train + 1) + '.pt')
            #     print('Load model_' + latest_model)
            #     model.fit(dataset, 
            #         eval_episodes=dataset, 
            #         n_steps_per_epoch = 10,
            #         n_steps = 10, 
            #         scorers={
            #                     'td_error': td_error_scorer,
            #                     'value_scale': average_value_estimation_scorer,
            #                 }

            #     )
                
            #     # model.save_model('saved_model_eval/model_' + str(i) + '.pt')
            #     saved_models = glob.glob('d3rlpy_logs/*')
            #     latest_model = max(saved_models, key=os.path.getctime)
            #     os.rename(latest_model, 'd3rlpy_logs/' + str(total_num_train + 1))
            
            total_num_train += 1
            print('There are ' + str(total_num_train) + ' being trained')
        except:
            total_num_train += 1
       
 
        



