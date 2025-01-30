import matplotlib.pyplot as plt
import random
import torch
import numpy as np

max_timestep=1000
min_timestep=0
num_timestep = max_timestep - min_timestep

steps=1000000
torch.manual_seed(1234)
# #a = [ int(random.random()*1000) for i in range(500)]
# sigma=20
# timesteps1 = ((torch.randn((steps,)).clip(-sigma, sigma) + sigma) / (2*sigma)) * (max_timestep - min_timestep) + min_timestep
# mu=500

# timesteps2 = mu + sigma * torch.randn((steps,))


timesteps3=torch.randint(low=min_timestep,high=max_timestep,size=(steps,)).long()

t = torch.sigmoid(0.9 * torch.randn((steps,)))
timesteps4 = (t*num_timestep).long()

t = torch.sigmoid(1.0 * torch.randn((steps,)))
timesteps5 = (t*num_timestep).long()

print(timesteps4)
print(timesteps5)

# plt.hist(timesteps3,1000,bins=1000, range=(0, 999), color='blue')
plt.hist(timesteps4,bins=1000, range=(0, 999), color='red')
plt.hist(timesteps5,bins=1000, range=(0, 999), color='black')
plt.show()