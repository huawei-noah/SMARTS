import os
import pickle
import numpy as np
import random

data_size = len(os.listdir('collected_data'))

index = random.randint(1, data_size)
index = 1
filename = str(index) + '-EGO.pkl'
with open('collected_data/' +  filename, 'rb') as f:
    data = pickle.load(f)

print(data[1]['actions'])