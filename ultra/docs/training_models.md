# Training existing models

ULTRA has supported the process to seperately **evaluate** saved models through the `evaluate.py` script. 
However, ULTRA has now extended support to **training** of previously saved models. This will allow users to 
continue training their saved models from a previous experiment. 

To accomodate this feature the experiment directory must include the **model** needed to be further trained and
the **replay buffer** used by the agent. 

> Although, it is not necessary to save the replay buffer inside the experiment directory, the performance of training the model will be affected due to no prior experience replay.

As an example, we have trained a SAC agent for 10000 episodes and at every 200th episodic checkpoint a model is saved.
The replay buffer is also saved at the end of the experiment. The experiment directory has been stored in our local
directory. By the running the following command, we can train a single model from the experiment directory. 
```sh
    python ultra/train.py --task 1 --level easy --episodes 10 --experiment-dir <path-to-experiment-dir>
```

> We have chosen to train only one model from the experiment directory to ensure simplicity. If there are multiple models inside the experiment directory then the last model which was saved will be used.
