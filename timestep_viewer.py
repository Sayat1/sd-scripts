import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from library.train_util import get_timesteps

max_timestep=400
min_timestep=0
num_timestep = max_timestep - min_timestep

steps=1000000
generator = torch.manual_seed(1234)

class dummy():
    timestep_sampling = "uniform"
    sigmoid_scale = 1.0
    sigmoid_bias = 0.0

args = dummy()
args.timestep_sampling = "sigmoid"
args.sigmoid_scale = 1.0
args.sigmoid_bias = 0.0

timesteps2 = get_timesteps(args,min_timestep,max_timestep,steps)

args.sigmoid_bias = -0.2
timesteps3 = get_timesteps(args,min_timestep,max_timestep,steps)

plt.hist(timesteps2,bins=1000, range=(0, 999), color='blue')
plt.hist(timesteps3,bins=1000, range=(0, 999), color='red')
plt.xticks(np.arange(0, 1000, step=100))
plt.show()