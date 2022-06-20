import pickle
import numpy as np

with open('/home/kyber/SMARTS/examples/baseline/online_cruise/EGO.pkl', 'rb') as f:
    replay_data = pickle.load(f)
    print(replay_data[19.3]['obs'].top_down_rgb.data)
    print(replay_data[19.3]['rewards'])
