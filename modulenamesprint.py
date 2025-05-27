from diffusers import DiffusionPipeline
from library import sdxl_model_util, sdxl_train_util
from library.train_util import get_hidden_states_sdxl
from library.custom_train_functions import get_weighted_text_embeddings_sdxl
import torch
import os
from torch import tensor
import time

class dummy:
    tokenizer_cache_dir = None
    max_token_length = None

max_token_length = 150

dummy_args = dummy()
dummy_args.max_token_length = max_token_length
sdxl_chekpoint_path = r"H:\ComfyUI_windows_portable\ComfyUI\models\Stable-diffusion\cyberrealisticPony_v10.safetensors"

tokenizers = sdxl_train_util.load_tokenizers(dummy_args)
model_version = sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0

text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info = sdxl_model_util.load_models_from_sdxl_checkpoint(model_version, sdxl_chekpoint_path, "cpu", torch.float16, None)

te1_names=[]
te2_names=[]
unet_names=[]

count=0

for name, module in text_encoder1.named_modules():
    te1_names.append({name:module.__class__.__name__})

for name, module in text_encoder2.named_modules():
    te2_names.append({name:module.__class__.__name__})

for name, module in unet.named_modules():
    unet_names.append({name:module.__class__.__name__})

folder = r"H:\kohya_ssPR\clipoutput"
with open(os.path.join(folder,"te1_module_names.txt"),mode='w',encoding='utf-8') as f:
    for names in te1_names:
        for key,value in names.items():
            f.write(f"{key} | {value}\n")

with open(os.path.join(folder,"te2_module_names.txt"),mode='w',encoding='utf-8') as f:
    for names in te2_names:
        for key,value in names.items():
            f.write(f"{key} | {value}\n")

with open(os.path.join(folder,"unet_module_names.txt"),mode='w',encoding='utf-8') as f:
    for names in unet_names:
        for key,value in names.items():
            f.write(f"{key} | {value}\n")