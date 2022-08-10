# Offline Dataset
The `SMARTS/competition/offline_dataset` directory has the following structure.
```text
offline_dataset                
    ├── <scenario_id>                # each scene in tfrecord
    |   ├── <time>_<vehicle_id>.png  # ego-centric bird-eye view image
    |   |  .
    |   |  .
    |   |  <vehicle_id>.pkl          # state space of the vehicle
    ├── <scenario_id>
    |   .
    |   .
```

# Examples
Two examples are provided for Track 2 offline learning based model development.
+ **The policy here has not yet been trained to fully solve the task environments.** 
+ **This example is only meant to demonstrate one potential method of developing a offline learning using Waymo dataset. Here, any offline learning method may be used to develop the policy.**

# Example 1 - Conservative Q-Learning
This example uses Convervative Q-learning (CQL) method from [d3rlpy](https://github.com/takuseno/d3rlpy) offline RL library.

## Setup
+ Use `python3.8` to develop your model.
    ```bash
    $ cd <path>/SMARTS/competition/track2/train
    $ python3.8 -m venv ./.venv
    $ source ./.venv/bin/activate
    $ pip install --upgrade pip
    $ pip install -e .
    ```
+ SMARTS is used as a dependent package.

## Notes on Observation, Action, and Reward
+ Observations: We use a 3-channel rgb birds eye view image plus an extended channel containing the location of the goal as the observation training. So the observation is of the form (4, 256, 256)
+ Actions: The action space (output of the policy) is using dx, dy and dh, which are the value change per step in x, y direction and heading for the ego vehicle in its birds eye view image coordinate. Since dx and dy can not be directly obtained from smarts observation, we have to get displacement change in global coordinate first and use a rotation matrix w.r.t the heading to get dx, dy.
+ Rewards: The reward use the default reward in SMARTS which is the distance travelled per step plus an extra reward for reaching the goal. Since there is not a "goal" concept in the training set, we use the last point of each trajectory as the goal position for training. 

## Train
1. Train
    ```bash
    $ cd <path>/SMARTS/competition/track2/train
    $ python3.8 train_cql.py --input_dir=<path_to_data> --ouput_dir=<path_to_saved_model>
    ```

    The default value for `input_dir` and `output_dir` are `/offline_dataset` and `/output`
 1. Since we can not load too many images in the training dataset at each time, we are training using data in one scenario at each time. After the end of each training iteration, the model will be saved in `<path>/SMARTS/competition/track2/train/d3rlpy_logs/<scenario_index>`. The next trainig iteration will keep training on the latest trained model. And at the end of the training, the last model will be copied to `/output`.

# Example 2 - Imitation Learning
## Data
+ Observations: We use a 3-channel rgb birds eye view image of the form (3, 256, 256) plus oridented and normalized dx & dy between the position of the ego vehicle and the goal location at each time step. dx & dy are calculated by first orienting both the current position and the goal location with respect to the current heading then substracting the oriented current position from the oriented goal location. dx & dy are then normalized using MinMaxScaler whose bound is (-0.1, 0.1).
+ Actions: The action space (output of the policy) is using dx, dy and dh, which are the value change per step in x, y direction and heading for the ego vehicle in its birds eye view image coordinate. dh is normalized by multiplying the values by 100. Since dx and dy can not be directly obtained from smarts observation, we have to get displacement change in global coordinate first and use a rotation matrix w.r.t the heading to get dx, dy. In evaluation, the values of predicted dh need to be divided by 100.
+ Rewards: The reward use the default reward in SMARTS which is the distance travelled per step plus an extra reward for reaching the goal. Since there is not a "goal" concept in the training set, we use the last point of each trajectory as the goal position for training. 

## Train
1. Train
    ```bash
    $ python train_IL.py --dataset_path <path_to_data> \
                        --output_path <path_to_saved_model> \
                        [--cache] False \
                        [--learning_rate] 0.001 \
                        [--save_steps] 10 \
                        [--batch_size] 32 \
                        [--num_epochs] 100 \
    ```
1. First time running `train_IL.py`, please set `cache=False`, the processed data will be saved to `./output/dataset.npy`. For later use, set `cache=True` and it will use the cached dataset.