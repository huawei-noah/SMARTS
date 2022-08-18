# Game of Tag
This directory contains a a multi-agent adversarial training demo. In the demo, there is a predator vehicle and a prey vehicle.
The predator vehicle's goal is to catch the prey, and the prey vehicle's goal is to avoid getting caught. 

## Run training
python examples/game_of_tag/game_of_tag.py examples/game_of_tag/scenarios/game_of_tag_demo_map/

## Run checkpoint
python examples/game_of_tag/run_checkpoint.py examples/game_of_tag/scenarios/game_of_tag_demo_map/

## Setup:
### Rewards
The formula for reward is 0.5/(distance-COLLIDE_DISTANCE)^2 and capped at 10

- COLLIDE_DISTANCE is the observed distance when two vehicle collides. Since the position of two vehicle is at the center, the distance when collesion happens is not exactly 0. 

### Common Reward:
    Off road: -10

#### Prey:
    Collision with predator: -10
    Distance to predator(d): 0.5/(d-COLLIDE_DISTANCE)^2 
#### Predator:
    Collision with predator: -10
    Distance to predator(d): 0.5/(d-COLLIDE_DISTANCE)^2 

### Action:
Speed selection in m/s: [0, 3, 6, 9]

Lane change selection relative to current lane: [-1, 0, 1]

## Output a model:
Currently Rllib does not have implementation for exporting a pytorch model. 

Replace `export_model`'s implementation in `ray/rllib/policy/torch_policy.py` to the following:
```
torch.save(self.model.state_dict(),f"{export_dir}/model.pt")
```
Then follow the steps in game_of_tag.py to export the model.

## Possible next steps
- Increase the number of agents to 2 predators and 2 prey. 
This requires modelling the reward to still be a zero sum game. The complication can be understood from 
how to model the distance reward between 2 predators and 1 prey. If the reward is only from nearest predator 
to nearest prey, the sum of predator and prey rewards will no longer be 0 because 2 predators will be getting full 
reward from 1 prey but the prey will only get full reward from 1 predator. This will require the predators to know about each 
other or the prey to know about other prey, and the prey to know about multiple predators.
- Add an attribute in observations to display whether the ego car is in front of the target vehicle or behind it, this may 
help to let ego vehicle know whether it should slow down or speed up