# Training existing models

ULTRA has supports the process of seperately **evaluating** saved models through the
`evaluate.py` script. ULTRA now extends support to **training** of previously saved
models. This will allow users to continue training their saved models from a previous
experiment. 

To use this feature, an existing experiment directory must be specified. Training of
each agent of this experiment will resume from the most recent model(s) for each agent.

> The baseline agents save data that is needed for the resumption of training at the end
of a training run. This data is saved in the experiment directory under the
[`extras` directory](getting_started.md#training-a-baseline-agent). An example of this
data includes the agent's replay buffer experience.

As an example, say we have trained a SAC agent for 10000 episodes where a model
checkpoint is saved every 200 episodes. The agent's replay buffer is also saved at the
end of the experiment. After the experiment is complete, the experiment directory has
been stored in our local directory. By the running the following command, we can resume
training of the SAC agent starting from the most recent model checkpoint and utilizing
the replay buffer experience gathered from the previous training run.

```sh
$ python ultra/train.py --task 1 --level easy --episodes 10 --experiment-dir <path-to-experiment-dir>
```
