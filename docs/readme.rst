.. _starter_kit:

Starter Kit
===========

===============
1. Introduction
===============

Scalable Multi-Agent RL Training School (**SMARTS**) is an autonomous driving platform for RL.

This starter kit helps you set up the develop environment, get familiar with SMARTS, write your RL training algorithms, monitor
your training process, evaluate your training results, and submit solutions to the SMARTS challenge on CodaLab.

========
2. Setup
========

To setup SMARTS to run simulations, see the commands in `Setup <doc/sections/Setup.md>`_.


===============
3. About SMARTS
===============

SMARTS supports fast and flexible construction of RL environments and flexible scripting of training scenarios in Python.

3.1 Environment
---------------

SMARTS provide two kinds of training environments, one is `HiwayEnv` following the convention of `Open AI Gym <https://gym.openai.com/`_ and another is `RLlibHiwayEnv` customized for `RLlib <https://docs.ray.io/en/master/rllib.html>`_ training.
To learn about these environments, see `SMARTS Env <doc/sections/Env.md>`_.

3.2 Scenario Studio
-------------------

Scenario Studio is a flexible tool to help you generate different traffics in a scenario. You use the Scenario Studio to add social vehicles with
different attributes, generate routes file for social vehicles and missions file for agents relatively.
See `Scenario Studio <sections/Sstudio.md>`_ for details.

.. image:: _static/smarts_envision.gif

====================================
4. Build Your First Training Program
====================================

4.1 Prepare the RL env
----------------------

4.1.1 Customize your map through Scenario Studio
------------------------------------------------

We have provided some public scenarios in dataset_public.

To add social vehicles and assign different attributes to them, you can refer to `Scenario Studio <sections/Sstudio.md>`_ session.
An example named `scenario/scenario.py` have been provided to you.

To improve the generalizability of your trained models to new maps, you may want to consdier creating your own maps and use them with SMARTS for training. See `Scenario Studio <sections/Sstudio.md>`_ for more.

4.1.2 Create the RL environment
-------------------------------

As mentioned before in `SMARTS Env <doc/sections/Env.md>`_, SMARTS provide two types of `Env` Class. `RLlibHiwayEnv` is designed for RLlib framework, which is
efficient and scalable but somewhat complex. For participants who do not want to use RLlib, you can use `HiwayEnv` to run your simulations. Whereas `HiwayEnv`
follows the `gym.env` style, `RLlibHiwayEnv` inherits from `RLlib.env.MultiAgentEnv`.

4.2 Create agents
-----------------

The `Agent` class is the interface between user code and a SMARTS environment. In `Agent`, you can specify the `AgentInterface` parameters to set the
observation information you need and the `ActionType`.

Also, you can create and use `adapters` to wrap interaction between an agent model and a SMARTS environment. Thesea adapters can transform observations from a SMARTS environment before using them as input to your model or transform your model's output before sending them as actions to SMARTS environment.

See `Agent <doc/sections/Agent.md>`_ for more detail.


4.3 Create policy and begin training
------------------------------------

We provide two examples to get your started with experimenting. “Random” is a random agent policy to demonstrate
what the most minimal solution could be and it uses `HiwayEnv`. “RLlib” demonstrates a complete solution using `RLlib <https://ray.readthedocs.io/en/latest/rllib.html>`_
to train a policy using `PPO <https://openai.com/blog/openai-baselines-ppo/>`_ and it uses `RLlibHiwayEnv`.

4.3.1 Random Example
--------------------

To run the random agent,

.. code-block:: bash

    python3 ~/src/starter_kit/random_example/run.py

The output should look something like this,

.. code-block:: bash

    Retrying in 1 seconds
    ...
    simulation ended
    Accumulated reward: 11.999999999999993

To use the SMARTS simulation for training, you can just create an instance of `HiwayEnv` as how you would create a normal Gym env and continue with your training.

4.3.2 RLlib Example
-------------------

Bundled in the starter kit is a more sophisticated example demonstrating how to use the `RLib <https://ray.readthedocs.io/en/latest/rllib.html>`_ framework to train an agent.
**Refer** to `RLlib <doc/sections/RLlib.md>`_ session for quick start and some tips.

RLlib supports both TensorFlow and PyTorch. It provides features for easily setting up and running experiments across many distributed nodes or across all CPU’s on a single node.

The RLlib example consists of 3 files, the `agent.py`, `trainer.py` and `run.py`, and a `model` directory.

* `agent.py` defines
    * the observation/reward/action adapter functions,
    * the model architecture,
    * and some code to build a Policy class that we can use for evaluating the trained model.
* `example_trainer.py` demonstrates how to setup an RLlib experiment and train a RL agent with example hyperparameter.
* `pbt_trainer.py` demonstrates how to setup an RLlib experiment and use pbt algorithm to train a RL agent.
* `run.py` uses the Policy class defined in agent.py to evaluate the trained model.
* `model/` contains a pre-trained network that was generated by pbt_trainer.py.

To train a new model, first backup the current model in rllib_example/model and then run:

.. code-block:: bash

    python3 ~/src/starter_kit/rllib_example/trainer.py ~/src/dataset_public/3lane
    # replace 3lane with the scenario to train against

This will train a new model and output it to the rllib_example/model/ directory.


4.4 Evaluate a model
--------------------

4.4.1 Random Example
--------------------

Similar to mentioned before, see `random_example/run.py` for more details.

4.4.2 Rllib example
-------------------

To evaluate a model:

.. code-block:: bash

    Python3 ~/src/starter_kit/rllib_example/run.py


4.5 visualization
-----------------

To monitor your training performance and some metrics, RLlib tensorboard log is by default stored at `/home/ray_results`.
Try command:

.. code-block:: bash

    tensorboard --logdir /home/ray_results

To see simulation rendering, refer to `Visualization <doc/sections/Visualization.md>`_ for details.

==================================
5. Codalab Datasets and Submission
==================================

5.1 Datasets
------------

Alongside the starter kit a public dataset is available on CodaLab. It’s up to you how to split this dataset up for training, testing, etc. Some or all of the following scenarios may be provided,

`3lane` will be provided in-built in the starter kit

* `1lane` - the simplest scenario, a one-lane loop
* `1lane_sharp` - a one-lane loop, with sharp curves
* `2lane_bwd` - a two-lane loop going backwards
* `2lane_sharp` - a two-lane loop with sharp curves
* `3lane` - a simple three-lane loop
* `3lane_bwd_b` - a simple three-lane loop going backwards with a different shape than `3lane`
* `3lane_sharp` - a three-lane loop with sharp curves
* `3lane_sharp_bwd_b` - a three-lane loop with sharp curves, going backwards, and a different shape than `3lane-sharp`

In addition to these scenarios, you can set the max step length and the random number generator seed in the `run.py` script. These settings could also be important for your experiments. You will want to consider using many episodes to continue training for longer periods of time. By default, the environment returns `done` when the agent is off road, gets into an accident, or hits the max step length and ends an episode.

All these settings together allow you to build larger and more varied setups to give your agent adequate learning experience for better performance.



5.2 Submission
--------------

When you submit your solution we will put it through an automated evaluation similar to your local `run.py script`. However we’ll be evaluating it across a different set of scenarios with different maps and varying numbers of social vehicles. We also run with different seed, max step count, and episode count.

When you’re happy with your solution and ready to submit to CodaLab for evaluation, you zip your policy (and any associated files) and upload to CodaLab under “Participate > Submit/View Files”. Important: zip together just the files, not a directory with the files in it. Be careful to make sure your solutions run locally, and perform well before submitting as the upload limit is fixed.

Your example submission zip dir structure can be like this:

.. code-block:: python

    - agent.py # defines agent so that codalab evaluation will import like from agent import agent
    - model/ # stores training model so that policy class in agent.py will restore from it.

The evaluation programm will act like below way:

.. code-block:: python

    from agent import agent
    agent.policy.setup()

    ...
    observation = observations[agent_id]
    actions = {agent_id: agent.act(observation)}
    observations, rewards, done, _ = env.step(actions)
    ...

    agent.policy.teardown()

Therefore in your agent class need to have the fields and methods mentioned before.

Building on top of the previous examples, or starting from scratch you can create your own policies by implementing the
`Policy` interface.

.. code-block:: python

    class Policy():
        def setup(self):
            # called once after import and can be used to load your model

        def teardown(self):
            # clean-up any resources

        def act(self, observation):
            # takes an observation, and returns an action


    model_path = Path(__file__).parent / "model"

    agent = Agent(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=1000),
        policy=Policy(model_path),
        observation_space=OBSERVATION_SPACE,
        action_space=ACTION_SPACE,
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
    )

5.3 Leaderboard
---------------

If your solution succeeds it will automatically get posted to the CodaLab leaderboard under the results tab. You will see your score across all the evaluation scenarios and a final rank score which is the sum of them all. This is the score based on which the winners will be chosen.
