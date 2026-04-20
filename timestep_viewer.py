import matplotlib.pyplot as plt
import random
import pytorch_optimizer.lr_scheduler
import torch
from functools import partial
import json
import numpy as np
import math
from diffusers import DDPMScheduler,EulerDiscreteScheduler
from library.train_util import get_timesteps
from  library import train_util

max_timestep=1000
min_timestep=0
num_timestep = max_timestep - min_timestep

steps=3000
#generator = torch.manual_seed(1234)

def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)

def apply_snr_weight(timesteps, noise_scheduler, gamma):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    snr_weight = torch.div(min_snr_gamma, snr).float().to("cpu")
    return snr_weight

def apply_snr_weight_medium(timesteps, noise_scheduler):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    snr_weight = torch.div(noise_scheduler.all_snr[500], snr).float().to("cpu")
    snr_weight = torch.clamp(snr_weight,min=0.01)
    return snr_weight

def debias_beta(beta: float, step: int) -> float:
    r"""Apply the Adam-style debias correction into beta.

    Simplified version of `\^{beta} = beta * (1.0 - beta ** (step - 1)) / (1.0 - beta ** step)`

    :param beta: float. beta.
    :param step: int. number of step.
    """
    beta_n: float = math.pow(beta, step)
    return (beta_n - beta) / (beta_n - 1.0)  # fmt: skip

noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
# prepare_scheduler_for_custom_training(noise_scheduler,"cpu")

# timesteps = noise_scheduler.timesteps[[0,500]].long().to(device="cpu")
# print(timesteps)
# print(noise_scheduler.all_snr[timesteps])

# euler_noise_scheduler = EulerDiscreteScheduler(
#         beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, timestep_spacing = "leading")
# prepare_scheduler_for_custom_training(euler_noise_scheduler,"cpu")
# timesteps = euler_noise_scheduler.timesteps[[0,500]].long().to(device="cpu")
# print(timesteps)
# print(euler_noise_scheduler.all_snr[timesteps])



# snr_dict = {}
# min_snr_dict = {}
# for t in range(1000):
#     snr_dict.update({t:noise_scheduler.all_snr[t].to("cpu").item()})

# for t in range(1000):
#     min_snr_dict.update({t:apply_snr_weight_medium([t],noise_scheduler).item()})

# with open(r"H:\ConvertScripts\snr.txt",mode='w',encoding='utf-8') as f:
#     json.dump(snr_dict,f,indent=2)

# with open(r"H:\ConvertScripts\snr_medium_norm.txt",mode='w',encoding='utf-8') as f:
#     json.dump(min_snr_dict,f,indent=2)

timesteps = list(range(0,1000,100))
timesteps += [1,999]
timesteps.sort()
timesteps_tensors = torch.tensor(timesteps)
scale_timesteps = noise_scheduler.alphas_cumprod[timesteps_tensors]
print(timesteps)
print(f"1:\t{[f"{v:.2f}" for v in (1-scale_timesteps)]}")
print(f"0.5:\t{[f"{v:.2f}" for v in (1-scale_timesteps)**0.5]}")
print(f"2:\t{[f"{v:.2f}" for v in (1-scale_timesteps)**10]}")


#((1-scale_timesteps)**1).view(-1, 1, 1, 1)


# class dummy():
#     train_scheduler = "pndm"
#     train_scheduler_args=None
#     timestep_sampling = "uniform"
#     sigmoid_scale = 1.0
#     sigmoid_bias = 0.0
#     timestep_shift=1.0
#     discrete_flow_shift=3
#     use_flow_timesteps = False


# args = dummy()
# # args.sigmoid_scale = 3 #작을수록 뾰족함
# # args.sigmoid_bias = 0.5 #클수록 우측으로감

# args.timestep_sampling = "sigmoid"
# args.sigmoid_scale = 1
# args.sigmoid_bias = -1
# t1=get_timesteps(args,min_timestep,max_timestep,steps,noise_scheduler)

# args.timestep_sampling = "sigmoid"
# args.sigmoid_scale = 0.6
# args.sigmoid_bias = 0
# t2=get_timesteps(args,min_timestep,max_timestep,steps,noise_scheduler)

# args.timestep_sampling = "sigmoid"
# args.sigmoid_scale = 1
# args.sigmoid_bias = 1
# t3=get_timesteps(args,min_timestep,max_timestep,steps,noise_scheduler)


# plt.hist(t1,bins=1000, range=(0, 999), color='red', alpha=0.6)
# plt.hist(t2,bins=1000, range=(0, 999), color='blue', alpha=0.6)
# plt.hist(t3,bins=1000, range=(0, 999), color='green', alpha=0.6)




# plt.xticks(np.arange(0, 1000, step=100))
# plt.show()