

I get that I need to choose an agent's policy and interface 
in an agent.py sort of file, 


Hi @atanas-kom,

On the question of "what about training and saving the agent, do I need to register it for everything to work?"

The answer depends

If you are interested in evaluating trained models in benchmarks (namely. `driving_smarts_2022` and `driving_smarts_2023.3`), the following applies. A benchmark is simply a set of environments with fixed scenarios, and scoring, which can be used to assess and compare the performance of agents built by various researchers.

Users are free to use any training method and any folder structure for training their policy. Only the inference code is required for evaluation in benchmark, and therefore it must follow the folder structure and contain specified file contents, as explained in the [docs](https://smarts.readthedocs.io/en/latest/benchmarks/driving_smarts_2023_3.html#code-structure). Agents need to be registrable in SMARTS, for them to be loaded and evaluated by the benchmarks. 

The example provided for `driving_smarts_2023.3` benchmark uses PPO algorithm from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) library to train a model. The model is trained by calling [model.learn](https://github.com/huawei-noah/SMARTS/blob/3d8b7b0da41020072a9bb9670388b41cc975cc8a/examples/rl/platoon/train/run.py#L145-L149) and saved using [checkpoint_callback](https://github.com/huawei-noah/SMARTS/blob/3d8b7b0da41020072a9bb9670388b41cc975cc8a/examples/rl/platoon/train/run.py#L130-L134) and a final [model.save()](https://github.com/huawei-noah/SMARTS/blob/3d8b7b0da41020072a9bb9670388b41cc975cc8a/examples/rl/platoon/train/run.py#L155) call.  

On the other hand, if you are only interested in training and evaluating models using SMARTS environments, you may do so directly as shown in the [intersection](https://smarts.readthedocs.io/en/latest/examples/intersection.html) example. Here, users are free to write their own training and evaluation code, with no restrictions. In this case, agents need not be registrable in SMARTS. Please note that the Colab in the intersection example is currently not usable.
