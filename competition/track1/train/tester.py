import gym
import numpy as np
if __name__ == "__main__":
    
    env = gym.make(
        "smarts.env:multi-scenario-v0",
        scenario='3lane_overtake',
        visdom=True
    )
    
    o = env.reset()
    init_pos = o['Agent_0'].ego_vehicle_state.position[:2]
    init_heading = o['Agent_0'].ego_vehicle_state.heading
    print(f'init_pos: {init_pos} init_heading: {init_heading:.2f}')
    
    a = np.array([init_pos[0] + 50, init_pos[1], init_heading + 1.57, 0.1])
    o, r, d, info = env.step({'Agent_0': a})
    print(f'action: {a}')
    pos = o['Agent_0'].ego_vehicle_state.position[:2]
    heading = o['Agent_0'].ego_vehicle_state.heading
    print(f'pos: {pos} heading: {heading:.2f}')
    print(f'done: {d["Agent_0"]}')
    
    pass