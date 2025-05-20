from diffusers import DiffusionPipeline
from library import sdxl_model_util, sdxl_train_util
from library.train_util import get_hidden_states_sdxl
from library.custom_train_functions import get_weighted_text_embeddings_sdxl
import torch
from torch import tensor
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class dummy:
    tokenizer_cache_dir = None
    max_token_length = None

max_token_length = 150

dummy_args = dummy()
dummy_args.max_token_length = max_token_length
sdxl_chekpoint_path = r"H:\ComfyUI_windows_portable\ComfyUI\models\Stable-diffusion\cyberrealisticPony_v10.safetensors"

tokenizers = sdxl_train_util.load_tokenizers(dummy_args)
model_version = sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0
text_encoder1, text_encoder2,vae, unet, ckpt_info, clip = sdxl_model_util.load_models_from_sdxl_checkpoint(model_version, sdxl_chekpoint_path, "cpu", torch.float16, None)

tokenizer = tokenizers[1]
text_encoder = text_encoder2

text_encoder.eval().cuda()

# Define tokens to compare
tokens = ["ahu","girl","solo","breasts","smile","high-waist shorts","simple background","rabbit toy","blonde hair","holding","closed mouth","jewelry","bare shoulders","standing","collar bone","swimsuit","cowboy shot","earrings","shorts","grey background","necklace","bracelet","holding microphone","croptop","lips","denim","ring","denimshorts","striped clothes"]
# 임베딩 저장용 리스트
embeddings = []
no_list = []

print(tokenizer.tokenize("ahu,"))       # 토큰 확인
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("ahu,")))
torch.detach


exit()

###############바꾸기############33
token = "ahu"
token_id = tokenizer.convert_tokens_to_ids(token)
print(text_encoder.get_input_embeddings().weight[token_id])
from torch.nn.functional import normalize
# 무작위 벡터 초기화 (normalize to match scale)
new_embedding = normalize(torch.randn((1, text_encoder.config.hidden_size)), dim=-1)

# 임베딩 레이어에 적용
with torch.no_grad():
    text_encoder.get_input_embeddings().weight[token_id] = new_embedding
###############바꾸기end############33

token = "ahu"
token_id = tokenizer.convert_tokens_to_ids(token)
print(token_id)
print(text_encoder.get_input_embeddings().weight[token_id])