# Training existing models

Inside ULTRA, we are able to seperately **evaluate** saved models through the
`evaluate.py` script. ULTRA now extends the support to **train** those previously saved
models as well. Thus allowing users to continue training their models from previous
experiments. 

To use this feature, an existing experiment directory must be specified. 

> Only the latest model from each of the older agent(s) (i.e. from previous experiment) is 
chosen to intialize the newer agent(s). This means that the new experiment will begin with
the agent(s) having an already experienced network.

> The baseline agents saves data that is needed for the resumption of training at the end
of a training run. This data is saved in the experiment directory under the
[`extras` directory](getting_started.md#training-a-baseline-agent). An example of this
data includes the agent's replay buffer experience.

As an example, say we have trained a SAC agent for 10000 episodes where a model
checkpoint is saved every 200 episodes. The agent's replay buffer is also saved at the
end of the experiment as well. After the experiment is complete, the experiment directory has
been stored in our local directory. By running the following command, we can resume
training of the SAC agent starting from the most recent model checkpoint and utilize
the replay buffer experience gathered from the previous training run.

```sh
$ python ultra/train.py --task 1 --level easy --episodes 10 --experiment-dir <path-to-experiment-dir>
```
