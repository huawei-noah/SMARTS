from remote_operations import remote_operations
import pickle
import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL, BCQ
import os
from PIL import Image
import re
import torch
torch.cuda.empty_cache()
import pathlib
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import td_error_scorer
import glob

prediction_step = 1
# tfrecord-00006-of-01000/ contain scenarios
path = '/net/storage-1/home/x50023223/smarts/examples/baseline/waymo_bev/tfrecord-00006-of-01000/' 
# remote_store_path= '/net/storage-1/home/c84201475/saved_model'
local_store_path = '/home/kyber/SMARTS/examples/baseline/waymo_offline/saved_model/'

remote = remote_operations()
client = remote.connect("10.193.241.239", "c84201475", "password") # gx2, username, password

scenarios = list()
for scenario_name in client.listdir(path):
    scenarios.append(scenario_name)




if not os.listdir('d3rlpy_logs/'):  # if empty
    index = 0
else:
    index = len(os.listdir('d3rlpy_logs/'))
    # existing_index = list()
    # for model_name in client.listdir(remote_store_path):
    #     existing_index.append(int(re.search('model_(.*).pt', model_name).group(1)))
    # index = sorted(existing_index)[-1] + 1



for scenario in scenarios[0:2]:
    obs = list()
    actions = list()
    rewards = list()
    terminals = list()
    print('processing scenario ' + scenario)
    vehicle_ids = list()
    
    for filename in client.listdir(path + scenario):
        if filename.endswith('.png'):
            vehicle_id = re.search('vehicle-(.*).png', filename).group(1)
            if vehicle_id not in vehicle_ids:
                vehicle_ids.append(vehicle_id)

    for id in vehicle_ids:
        print('adding data for vehicle id ' + id + ' in scenario ' + scenario)

        with client.file(path + scenario +  '/Agent-history-vehicle-' + id + '.pkl', 'rb') as f:
            vehicle_data = pickle.load(f)
        image_names = list()

        for filename in client.listdir(path + scenario):
            if filename.endswith('-' + id + '.png'):
                image_names.append(filename)

        image_names = sorted(image_names)

        for i in range(len(image_names) - 1):
            imgfile = client.open(path + scenario + '/' + image_names[i], 'r')
            imgfile.seek(0)
            image = Image.open(imgfile)
            # image = Image.open(path + scenario + '/' + image_names[i])

            sim_time = image_names[i].split('_Agent')[0]
            sim_time_next = image_names[i + 1].split('_Agent')[0]
            current_position = vehicle_data[float(sim_time)]['ego']['pos']
            next_position = vehicle_data[float(sim_time_next)]['ego']['pos']
            dx = next_position[0] - current_position[0]
            dy = next_position[1] - current_position[1]
            events = vehicle_data[float(sim_time)]['events']
            if all(value == 0 for value in events.values()):
                terminal = 0
            else:
                terminal = 1
            obs.append(np.asarray(image).reshape(3,256,256))
            actions.append([dx, dy])
            rewards.append(vehicle_data[float(sim_time)]['dist'])
            terminals.append(terminal)
        print(str(len(obs)) + ' pieces of data are added into dataset.' )

    obs = np.array(obs)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    dataset = MDPDataset(obs, actions, rewards, terminals)

    if index == 0:
        model = d3rlpy.algos.CQL(use_gpu=True, batch_size=1)
    else:
        saved_models = glob.glob('d3rlpy_logs/*')
        latest_model = max(saved_models, key=os.path.getctime)
        model = CQL.from_json('d3rlpy_logs/1/params.json', use_gpu=True)
        model.load_model(latest_model + '/model_1.pt')

    model.fit(dataset, 
            eval_episodes=dataset, 
            n_steps_per_epoch = 1,
            n_steps = 1, 
            scorers={
                        'td_error': td_error_scorer,
                        'value_scale': average_value_estimation_scorer,
                    }
        )

    saved_models = glob.glob('d3rlpy_logs/*')
    latest_model = max(saved_models, key=os.path.getctime)
    os.rename(latest_model, 'd3rlpy_logs/' + str(index + 1))
    # model.save_model(local_store_path + 'model_' + str(index) + '.pt')
    # client.put(local_store_path + 'model_' + str(index) + '.pt', remote_store_path + 'model_' + str(index) + '.pt')
    index += 1

imgfile.close()
client.close()
print("Finish Processing")

