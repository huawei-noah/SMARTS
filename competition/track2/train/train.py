import os
import argparse
from pathlib import Path
import sys
import subprocess

def train(path):
    from utility import goal_region_reward
    from utility import get_goal_layer
    from utility import get_trans_coor
    import pickle
    import numpy as np
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.preprocessing import MinMaxActionScaler
    from d3rlpy.algos import CQL, BCQ
    from PIL import Image
    import re
    import glob
    import shutil

    d3rlpy.seed(313)

    scenarios = list()
    for scenario_name in os.listdir(path):
        scenarios.append(scenario_name)


    if not os.path.isdir("d3rlpy_logs/"):
        index = 0
        os.mkdir('d3rlpy_logs')
    else:
        index = len(os.listdir("d3rlpy_logs/"))


    for scenario in scenarios[index : len(scenarios)]:
        try:
            obs = list()
            actions = list()
            rewards = list()
            terminals = list()
            print("processing scenario " + scenario)
            vehicle_ids = list()

            for filename in os.listdir(path + scenario):
                if filename.endswith(".png"):
                    match = re.search("vehicle-(.*).png", filename)
                    assert match is not None
                    vehicle_id = match.group(1)
                    if vehicle_id not in vehicle_ids:
                        vehicle_ids.append(vehicle_id)
            
            for id in vehicle_ids:
                print("adding data for vehicle id " + id + " in scenario " + scenario)

                with open(
                    path + scenario + "/Agent-history-vehicle-" + id + ".pkl", "rb"
                ) as f:
                    vehicle_data = pickle.load(f)
                image_names = list()

                for filename in os.listdir(path + scenario):
                    if filename.endswith("-" + id + ".png"):
                        image_names.append(filename)

                image_names = sorted(image_names)
                
                goal_pos_x = vehicle_data[float(image_names[-1].split("_Agent")[0])]["ego"][
                    "pos"
                ][0]
                goal_pos_y = vehicle_data[float(image_names[-1].split("_Agent")[0])]["ego"][
                    "pos"
                ][1]
                threshold = 3
                
                for i in range(len(image_names) - 1):
                    with Image.open(
                        path + scenario + "/" + image_names[i], "r"
                    ) as image:
                        image.seek(0)
                        sim_time = image_names[i].split("_Agent")[0]
                        sim_time_next = image_names[i + 1].split("_Agent")[0]
                        current_position = vehicle_data[float(sim_time)]["ego"]["pos"]
                        current_heading = vehicle_data[float(sim_time)]["ego"]["heading"]
                        next_position = vehicle_data[float(sim_time_next)]["ego"]["pos"]
                        next_heading = vehicle_data[float(sim_time_next)]["ego"]["heading"]
                        trans_coor = get_trans_coor(
                            next_position[0], 
                            next_position[1], 
                            current_position[0], 
                            current_position[1], 
                            current_heading
                        )
                        trans_cur = trans_coor[0]
                        trans_next = trans_coor[1]
                        dx = trans_next[0, 0] - trans_cur[0, 0]
                        dy = trans_next[1, 0] - trans_cur[1, 0]
                        dheading = next_heading - current_heading
                        events = vehicle_data[float(sim_time)]["events"]
                        if all(value == 0 for value in events.values()):
                            terminal = 0
                        else:
                            terminal = 1

                        bev = np.moveaxis(np.asarray(image), -1, 0)
                        goal_obs = get_goal_layer(
                            goal_pos_x,
                            goal_pos_y,
                            current_position[0],
                            current_position[1],
                            current_heading,
                        )
                        extended_ob = np.concatenate((bev, goal_obs), axis=0)
                        obs.append(extended_ob)
                        actions.append([dx, dy, dheading])
                        dist_reward = vehicle_data[float(sim_time)]["dist"]
                        goal_reward = goal_region_reward(
                            threshold,
                            goal_pos_x,
                            goal_pos_y,
                            current_position[0],
                            current_position[1],
                        )
                        rewards.append(dist_reward + goal_reward)
                        rewards.append(dist_reward)

                        terminals.append(terminal)

                print(str(len(obs)) + " pieces of data are added into dataset.")

            obs = np.array(obs, dtype=np.uint8)
            actions = np.array(actions)
            rewards = np.array(rewards)
            terminals = np.array(terminals)
            dataset = MDPDataset(obs, actions, rewards, terminals)

            if index == 0:
                minimum = [-0.1, 0, -0.1]
                maximum = [0.1, 2, 0.1]
                action_scaler = MinMaxActionScaler(minimum=minimum, maximum=maximum)
                model = d3rlpy.algos.CQL(use_gpu=True, batch_size=32, action_scaler=action_scaler)
            else:
                saved_models = glob.glob("d3rlpy_logs/*")
                latest_model = max(saved_models, key=os.path.getctime)
                model = CQL.from_json("d3rlpy_logs/1/params.json", use_gpu=True)
                model.load_model(latest_model + "/model_100.pt")
            model.fit(
                dataset,
                eval_episodes=dataset,
                n_steps_per_epoch=100,
                n_steps=100,
            )
            saved_models = glob.glob("d3rlpy_logs/*")
            latest_model = max(saved_models, key=os.path.getctime)
            os.rename(latest_model, "d3rlpy_logs/" + str(index + 1))
            index += 1
        except:
            pass
    saved_models = glob.glob("d3rlpy_logs/*")
    latest_model = max(saved_models, key=os.path.getctime)
    os.rename(latest_model, "d3rlpy_logs/model")
    shutil.copytree('d3rlpy_logs/model', '../submission/model')



def main(args: argparse.Namespace):
    path = args.input_dir
    train(path)


if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--input_dir",
        help="The path to the directory containing the offline training data",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    # Install requirements.
    req_file = os.path.join(str(Path(__file__).absolute().parent), "requirements.txt")
    sys.path.insert(0, str(Path(__file__).absolute().parent))
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-1",
        ]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

    main(args)

