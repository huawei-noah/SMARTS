## instructions for MA trainer

### Overview:

**framework**: This package will include the proposed 4 (or more) MARL frameworks
- `dect.py`: decentralized execution centralized training framework (finished)
- `dedt.py`: a fully decentralized MARL framework (not finished yet, just an initialization)
- `il.py`: independent MARL (not finished yet, just an initialization)
- `share.py`: parameter sharing (not finished yet, just an initialization)

**zoo**: This package includes some predefined MARL policies rely on RLLib
- `centralized_a2c.py`: an implementation of centralized a2c

    **tuned**: This package includes tuned parameter configs and tuned functions.
    - `callback.py`: callbacks in rllib for evalation some metrics.
    - `default_model.py`: default model in trainers.
    - `tuned_space.py`: tuned space choice and wrap functions.
    - `simple_space.py`: simple space choice and wrap functions.
        
    **tests**:
    - `test_speed.py`: test SMARTS speed
        
    **utils**:
    - `scenario.py`: scripts to generate scenario with different social cars. 
    - `utils.py`: utils functions including exploration etc.
        

**run**: This package includes some running experiments
currently support continuous action space algorithm `PPO` and discrete action
 space algorithm `PG`, `A2C`, `A3C`, `DQN`
- `run_cc_train.py`: run centralized control.
- `run_evaluation.py`:  run evaluate model performance including rendering, scores.  
   `python trainer/run/run_evaluation.py --load_type model --load_path trainer/trained_model/loop/model`
- `run_pbt_learning.py`: run pbt learning for our choosen space and function settings.
- `run_il.py`: run independent learning learning.
- `run_share.py`: run share parameter learning.
   `python trainer/run/(run_pbt_learning/il/share.py) --num_agents 1 --scenario loop
   --num_workers 4`
- `run_single.py`: run single agent learning.
 
### Instructions:
For `.py` files in examples, run, just use `python *.py`, pay attention to your python path.

Also, **social vehcles adding** have been moved to scenario studio, so you need to refer to that to add social vehcles.
