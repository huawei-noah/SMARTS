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

prediction_step = 1
path = '/home/kyber/SMARTS/examples/baseline/waymo_offline/extracted_waymo/'
working_dir = '/home/kyber/SMARTS/examples/baseline/waymo_offline/'



scenarios = list()
for scenario_name in os.listdir(path):
    scenarios.append(scenario_name)

if not os.listdir('saved_model'):
    index = 0
else:
    existing_index = list()
    for model_name in os.listdir('saved_model'):
        existing_index.append(int(re.search('model_(.*).pt', model_name).group(1)))
    index = sorted(existing_index)[-1] + 1


for scenario in scenarios[0:3]:
    obs = list()
    actions = list()
    rewards = list()
    terminals = list()
    print('processing scenario ' + scenario)
    vehicle_ids = list()
    for filename in os.listdir(path + scenario):
        
        if filename.endswith('.png'):
            vehicle_id = re.search('vehicle-(.*).png', filename).group(1)
            if vehicle_id not in vehicle_ids:
                vehicle_ids.append(vehicle_id)
    for id in vehicle_ids:
        print('adding data for vehicle id ' + id + ' in scenario ' + scenario)
        with open(path + scenario +  '/Agent-history-vehicle-' + id + '.pkl', 'rb') as f:
            vehicle_data = pickle.load(f)
        image_names = list()
        for filename in os.listdir(path + scenario):
            if filename.endswith('-' + id + '.png'):
                image_names.append(filename)
        image_names = sorted(image_names)
        for i in range(len(image_names) - 1):
            image = Image.open(path + scenario + '/' + image_names[i])
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



        # for i in range(len(image_names) - prediction_step):
        #     image_group = image_names[i : i + prediction_step]
        #     for image_name in image_group:
        #         image = Image.open(path + scenario + '/' + image_name)
        #         obs.append(np.asarray(image).reshape(3,256,256))
        #         sim_time = image_name[0:3]
        #         sim_time_next = str(float(sim_time) + 0.1)
        #         print(sim_time, sim_time_next)

    obs = np.array(obs)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    dataset = MDPDataset(obs, actions, rewards, terminals)
    if index == 0:
        model = d3rlpy.algos.CQL(use_gpu=True, batch_size=1)
    else:
        saved_folder = sorted(os.listdir(pathlib.Path(__file__).absolute().parent/'d3rlpy_logs/'))[-1]
        model = CQL.from_json('d3rlpy_logs/' + saved_folder + '/params.json', use_gpu=True)

    model.fit(dataset, 
            eval_episodes=dataset, 
            n_epochs = 1, 

    )
    model.save_model('saved_model/' + 'model_' + str(index) + '.pt')
    index += 1






     
    
    

# with open(path + 'Agent-history-vehicle-2287.pkl', 'rb') as f:
#     data = pickle.load(f)

# allfiles = os.listdir(path)
# files = [ fname for fname in allfiles if fname.endswith('.png')]

# name = files[1]

# actions = np.random.random((4,2))
# rewards = np.random.random(4)
# terminals = np.array([0, 1])
# dataset = MDPDataset(observations, actions, rewards, terminals)   

# image = Image.open(path + name)
# image.show()
# print(image.format)

# print(image.size)

# print(image.mode)
# data = np.asarray(image)
# print(data)