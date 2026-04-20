import torch
from library import strategy_sdxl, strategy_base, sdxl_model_util
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import load_file
from tqdm import tqdm
import torch.nn.functional as F

LORA_PATH = r"h:\ComfyUI_windows_portable\ComfyUI\models\Lora\test\chaeyoung-111-LOKR8_conv8-PD9ILL-cy5_realjf_wd_nbg-gart_girl-PDP_b9_99_wd1e4_eps1e8-CONST-UNET-LR1-b2r10-rn_drops01-rcorbg10-1M_SDXL-modalkh-15E.safetensors"
TOP_K = 50
device = "cuda"
checkpoint_path = r"H:\ComfyUI_windows_portable\ComfyUI\models\Stable-diffusion\ILL\perfectdeliberate_v90.safetensors"

tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(None, None)
strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()
strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

tokenizers = [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

text_encoder1,text_encoder2,vae, unet,logit_scale, ckpt_info = sdxl_model_util.load_models_from_sdxl_checkpoint(None,checkpoint_path,"cuda",dtype=torch.bfloat16)

vae = unet = logit_scale = skpt_info = None
del vae, unet,logit_scale, skpt_info

tokenizer_l, tokenizer_g = tokenizers
text_encoder_l, text_encoder_g = text_encoder1, text_encoder2
text_encoder_l = text_encoder_l.to(device)
text_encoder_g = text_encoder_g.to(device)

text_encoder1.eval()
text_encoder2.eval()

weights = load_file(LORA_PATH)

# ─── 키 그룹핑 (같은 레이어끼리 묶기) ──────────────────
def group_lokr_keys(weights):
    """
    예시 키 패턴:
    lora_unet_...attn2_to_k.lokr_w1
    lora_unet_...attn2_to_k.lokr_w2
    lora_unet_...attn2_to_k.lokr_w2_a   (분해된 경우)
    lora_unet_...attn2_to_k.lokr_w2_b   (분해된 경우)
    lora_unet_...attn2_to_k.lokr_t2     (tucker 분해 있을 경우)
    """
    layers = {}
    for key in weights:
        # lokr 키만
        if "lokr_" not in key:
            continue
        # cross-attention to_k, to_v만
        if "attn2" not in key:
            continue
        if not ("to_k" in key or "to_v" in key):
            continue
        
        # base 레이어명 추출 (접미사 제거)
        # e.g. "....to_k.lokr_w1" → "....to_k"
        base = key.rsplit(".lokr_", 1)[0]
        layers.setdefault(base, {})[key] = weights[key]
    
    return layers

def compute_lokr_delta(layer_dict, base_name):
    """
    LoKR의 ∆W 계산
    세 가지 케이스 처리:
      1. w1 + w2          → kron(w1, w2)
      2. w1 + w2_a + w2_b → kron(w1, w2_a @ w2_b)
      3. w1_a + w1_b + w2 → kron(w1_a @ w1_b, w2)  (드문 케이스)
    """
    def k(name): return f"{base_name}.lokr_{name}"
    
    has = lambda name: k(name) in layer_dict
    get  = lambda name: layer_dict[k(name)].float()

    # w1 결정
    if has("w1"):
        w1 = get("w1")
    elif has("w1_a") and has("w1_b"):
        w1 = get("w1_a") @ get("w1_b")
    else:
        return None

    # w2 결정
    if has("w2"):
        w2 = get("w2")
    elif has("w2_a") and has("w2_b"):
        w2 = get("w2_a") @ get("w2_b")
    else:
        return None

    # Tucker (t2) 처리
    if has("t2"):
        t2 = get("t2")
        # tucker 분해: einsum으로 재구성
        w2 = torch.einsum("i j k l, i r, j s -> r s k l", 
                           t2, w2, w1).reshape(
                           t2.shape[0]*w2.shape[0], -1)
        return w2

    # 일반 크로네커 곱
    # torch.kron: A⊗B
    delta = torch.kron(w1, w2).to(device)  # [out, in]
    return delta


# ─── ∆W 미리 계산 ────────────────────────────────────
grouped = group_lokr_keys(weights)
delta_weights = []

for base_name, layer_dict in grouped.items():
    delta = compute_lokr_delta(layer_dict, base_name)
    if delta is not None:
        delta_weights.append((base_name, delta))

print(f"분석할 LoKR ∆W 레이어 수: {len(delta_weights)}")

# 차원 확인 (디버깅용)
for name, dw in delta_weights[:5]:
    print(f"  {name}: {dw.shape}")


# 서브워드/특수토큰 필터 (선택적)
def is_valid_token(word):
    if word.startswith("<") and word.endswith(">"):
        return False  # <|startoftext|> 등
    if word.startswith("\x00"):
        return False
    if len(word) <= 2:
        return False
    return True


def get_sdxl_embed(word):
    """SDXL과 동일하게 CLIP-L + CLIP-G concat → 2048차원"""
    
    tokens1, tokens2 = tokenize_strategy.tokenize([word])
    with torch.no_grad():
        hidden_state1, hidden_state2, pool2 = text_encoding_strategy.encode_tokens(
            tokenize_strategy, [text_encoder_l,text_encoder_g], [tokens1, tokens2]
        )

    text_embedding = torch.cat([hidden_state1, hidden_state2], dim=-1)
    return text_embedding

# ─── vocab 스캔 ──────────────────────────────────────
# vocab은 두 토크나이저 중 하나 (거의 동일)
vocab = tokenizer_l.get_vocab()

scores = {}

valid_vocab = {w: tid for w, tid in vocab.items() if is_valid_token(w)}
print(f"유효 토큰 수: {len(valid_vocab)}")

#valid_vocab = ["reslistic","gart girl","apple"]

scores = {}
for word in tqdm(valid_vocab,desc="vocab 스캔"):
    token_embed = get_sdxl_embed(word)                          # [2048]

    token_score = 0.0
    valid_count = 0
    for _, dw in delta_weights:
        activation = F.linear(token_embed, dw)
        token_score += activation.norm().item()
        valid_count += 1

    scores[word] = token_score / valid_count if valid_count > 0 else 0.0


# ─── 결과 출력 ───────────────────────────────────────
#print(scores)
sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
max_score = sorted_scores[0][1]


print(f"\n{'='*55}")
print(f"LoKR 반응 상위 {TOP_K} 토큰")
print(f"{'='*55}")
for rank_i, (word, score) in enumerate(sorted_scores[:TOP_K], 1):
    if max_score > 0:
        bar = "█" * int(score / max_score * 25)
    else:
        bar = "█"
    print(f"{rank_i:3d}. {word:20s}  {score:8.4f}  {bar}")