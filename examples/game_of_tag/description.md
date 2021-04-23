# have 1 predator and 1 prey, collision turned on in DoneCreteria
Rewards:
    Common:
        offroad: -1.5
        onshoulder: -0.2 
        action changed: -0.002
        Collesion with vehicles of same type: -1.5
    Prey:
        collide with predator: -1 
        min distance to any predator: distance * 0.005 
        Survived til game Ended: 1

        # distance to prey: -k*1/d^2 should be dominant and cap it, Sum up to 0
    Predator:
        collide with prey: 1 
        Chasing:
            <= 20 meter straightly behind prey + 0.05 
            <= 20 meter straightly behind prey AND speed > prey: +0.05 
        Blocking:
            <= 10 meter straightly in front of prey: +0.03 
        Game Ended and there is still prey: -1

        # distance to prey: k*1/d^2 should be dominant and cap it, Sum up to 0, k = 5

Action:
    speeds: 5 levels: [0, 3, 6, 9, 12]
    lanechange: -1, 0, 1

Observation:
    ego_position:
    PredatorVehicles and PreyVehicles:
        relative_speed
        position
        distance to vehicle
        rel_lane_index: -4, -3, -2, -1, 0, 1, 2, 3, 4 ( - larger than current lane, + smaller than current lane index)


Problems:
Distance to road curb useful for laner?
Distance to road curb enough for continuous to nagivate road? # basic Lidar sensor
Training result not obvious the agent learned
Problem with using exported TF1 model - ULTRA suggests using PyTorch


Suggestion:
Change the goal to something simpler -  racing?


### Rllib code hijacking:
torch.save(self.model.state_dict(),f"{export_dir}/model.pt")


# Self observation: check your own vehicle info

# x < 1: log(-x+1)


Do bellman 

Reward for predator:
u_predatodDesr=f(observation)


f(observation):=
if predator_pos>prey_pos
use this action (speed,lane_change)
------------------------------------
Reward:
control regularization of reward
old_reward- k*(u_predator-f(observation))**2


predator_reward(obs):
if predeator_offset>prey_offset:
des_predator_speed=prey_speed



-k*(obs.predator_speed-des_predator_speedd)**2 k>0



old_reward

Prey reward:
-k*(obs.prey_speed-des_prey_speedd)**2
