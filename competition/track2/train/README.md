# Examples
1. Two examples are provided for Track 2 offline learning based model development.
   + The primary example is present in the `SMARTS/competition/track2/train` folder. It is based on Conservative Q-Learning. Only the primary example is fully developed and docker ready.
   + The secondary example is present in the `SMARTS/competition/track2/train/secondary_example` folder. It is based on Imitation Learning and it is not docker ready.
1. **The policy here has not yet been trained to fully solve the task environments.** 
1. **This example is only meant to demonstrate one potential method of developing a offline learning using Waymo dataset. Here, any offline learning method may be used to develop the policy.**

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
