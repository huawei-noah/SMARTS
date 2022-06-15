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

prediction_step = 1
path = '/home/kyber/SMARTS/examples/baseline/waymo_offline/extracted_waymo/'

scenarios = list()
for scenario_name in os.listdir(path):
    scenarios.append(scenario_name)

obs = list()
filename_with_vehicle_id = list()
for scenario in scenarios:
    # print('processing scenario' + scenario)
    vehicle_ids = list()
    for filename in os.listdir(path + scenario):
        
        if filename.endswith('.png'):
            vehicle_id = re.search('vehicle-(.*).png', filename).group(1)
            if vehicle_id not in vehicle_ids:
                vehicle_ids.append(vehicle_id)
    for id in vehicle_ids:
        print('adding data for vehicle id ' + id + ' in scenario ' + scenario)
        with open(path + scenario +  '/Agent-history-vehicle-' + vehicle_id + '.pkl', 'rb') as f:
            vehicle_data = pickle.load(f)
        image_names = list()
        for filename in os.listdir(path + scenario):
            if filename.endswith(id + '.png'):
                image_names.append(filename)
        image_names = sorted(image_names)

        for i in range(len(image_names) - prediction_step):
            image_group = image_names[i : i + prediction_step]
            for image_name in image_group:
                image = Image.open(path + scenario + '/' + image_name)
                obs.append(np.asarray(image).reshape(3,256,256))

obs = np.array(obs)
actions = np.random.random((len(obs),2))
rewards = np.random.random(len(obs))
terminals = np.random.random(len(obs))
print(str(obs.shape[0]) + ' pieces of data are added into dataset.' )
dataset = MDPDataset(obs, actions, rewards, terminals) 
cql = d3rlpy.algos.CQL(use_gpu=True, batch_size=1)

cql.fit(dataset, 
        eval_episodes=dataset, 
        n_epochs = 1, 

)





     
    
    

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