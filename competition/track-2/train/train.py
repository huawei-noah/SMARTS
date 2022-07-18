from utility import remote_operations
from utility import goal_region_reward
from utility import get_goal_layer
import paramiko
import getpass
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



path = '/net/storage-1/home/c84201475/waymo_bev/' 
remote = remote_operations()
ip_add = input("Server IP: ")
user_name = input("Username: ")
pswd = getpass.getpass("Password: ")

if ip_add == "gx1":
    ip_add = "10.193.241.237"
elif ip_add == "gx2":
    ip_add = "10.193.241.238"
elif ip_add == "gx3":
    ip_add = "10.193.241.239"

while True:
    try:
        client = remote.connect(ip_add, user_name, pswd) # ip, username, password
        break
    except paramiko.ssh_exception.AuthenticationException:
        print("Authentication Failed")
        pswd = getpass.getpass("Password: ")


scenarios = list()
for scenario_name in client.listdir(path):
    scenarios.append(scenario_name)


if not os.listdir('d3rlpy_logs/'):  # if empty
    index = 0
else:
    index = len(os.listdir('d3rlpy_logs/'))


for scenario in scenarios[index:len(scenarios)]:
    try:
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

            goal_pos_x = vehicle_data[float(image_names[-1].split('_Agent')[0])]['ego']['pos'][0]
            goal_pos_y = vehicle_data[float(image_names[-1].split('_Agent')[0])]['ego']['pos'][1]
            threshold = 0.2

            for i in range(len(image_names) - 1):
                imgfile = client.open(path + scenario + '/' + image_names[i], 'r')
                imgfile.seek(0)
                image = Image.open(imgfile)

                sim_time = image_names[i].split('_Agent')[0]
                sim_time_next = image_names[i + 1].split('_Agent')[0]
                current_position = vehicle_data[float(sim_time)]['ego']['pos']
                current_heading = vehicle_data[float(sim_time)]['ego']['heading']
                next_position = vehicle_data[float(sim_time_next)]['ego']['pos']
                next_heading = vehicle_data[float(sim_time_next)]['ego']['heading']
                dx = next_position[0] - current_position[0]
                dy = next_position[1] - current_position[1]
                dheading = next_heading - current_heading
                events = vehicle_data[float(sim_time)]['events']
                if all(value == 0 for value in events.values()):
                    terminal = 0
                else:
                    terminal = 1
                
                img_obs = np.asarray(image).reshape(3,256,256)
                goal_obs = get_goal_layer(goal_pos_x, goal_pos_y, current_position[0], current_position[1], current_heading)
                obs.append(np.concatenate((img_obs, goal_obs), axis=0))

                actions.append([dx, dy, dheading])

                dist_reward = vehicle_data[float(sim_time)]['dist']
                goal_reward = goal_region_reward(threshold, goal_pos_x, goal_pos_y, current_position[0], current_position[1])
                rewards.append(dist_reward + goal_reward)
                
                terminals.append(terminal)
            print(str(len(obs)) + ' pieces of data are added into dataset.' )

        obs = np.array(obs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        dataset = MDPDataset(obs, actions, rewards, terminals)

        if index == 0:
            model = d3rlpy.algos.CQL(use_gpu=True, batch_size=32)
        else:
            saved_models = glob.glob('d3rlpy_logs/*')
            latest_model = max(saved_models, key=os.path.getctime)
            model = CQL.from_json('d3rlpy_logs/1/params.json', use_gpu=True)
            model.load_model(latest_model + '/model_100.pt')

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
        os.rename(latest_model, 'd3rlpy_logs/' + str(index + 1))
        index += 1
    except:
        pass

imgfile.close()
client.close()
print("Finish Processing")

