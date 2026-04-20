from library import strategy_sdxl, strategy_base, sdxl_model_util
import random
import string
import torch
import torch.nn.functional as F
import itertools


checkpoint_path = r"Z:\Stable-diffusion\ILL\illustrij_v21.safetensors"

tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(None, None)
strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
tokenizers = [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

text_encoder1,text_encoder2,vae, unet,logit_scale, ckpt_info = sdxl_model_util.load_models_from_sdxl_checkpoint(None,checkpoint_path,"cuda",dtype=torch.bfloat16)

vae = unet = logit_scale = skpt_info = None
del vae, unet,logit_scale, skpt_info

tokenizer_l, tokenizer_g = tokenizers


def analyze_sdxl_trigger(candidates, tokenizers, text_encoders):
    """
    tokenizers: {'L': tokenizer_l, 'G': tokenizer_g}
    text_encoders: {'L': encoder_l, 'G': encoder_g}
    """
    results = []

    for word in candidates:
        word_data = {"word": word, "details": {}, "total_score": 0, "is_valid": True}
        
        for key in ['L', 'G']:
            tokenizer = tokenizers[key]
            encoder = text_encoders[key]
            
            # 1. 토큰화 및 1토큰 여부 확인
            tokens = tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) != 1:
                word_data["is_valid"] = False
                break
                
            token_id = tokens[0]
            
            # 2. 임베딩 매트릭스 추출 및 정규화
            embeddings = encoder.get_input_embeddings().weight.data
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # 3. 코사인 유사도 계산
            target_vec = norm_embeddings[token_id].unsqueeze(0)
            similarities = torch.matmul(target_vec, norm_embeddings.T).squeeze()
            
            # 자기 자신 제외 후 최대 유사도 추출
            similarities[token_id] = -1.0 
            max_sim, max_idx = torch.max(similarities, dim=0)
            closest_token = tokenizer.convert_ids_to_tokens([max_idx.item()])[0]
            
            # 결과 저장 (유사도가 낮을수록 점수가 높음)
            sim_val = max_sim.item()
            word_data["details"][key] = {
                "max_sim": sim_val,
                "closest": closest_token
            }
            word_data["total_score"] += (1 - sim_val)

        if word_data["is_valid"]:
            # 두 모델의 평균 점수로 최종 순위 결정
            word_data["total_score"] /= 2
            results.append(word_data)

    # 점수 높은 순 정렬
    results.sort(key=lambda x: x['total_score'], reverse=True)
    return results

def find_dual_one_token_candidates(tokenizer_l, tokenizer_g, min_len=3, max_len=6):
    # 1. 각 토크나이저의 단어 사전에서 조건에 맞는 토큰 추출
    # 특수 기호(</w>, ## 등)를 제외하고 순수 알파벳 조합만 추출합니다.
    
    def get_valid_tokens(tokenizer):
        valid_set = set()
        for token, token_id in tokenizer.get_vocab().items():
            # 토크나이저 특수 접미사 제거 (예: CLIP의 '</w>')
            clean_token = token.replace('</w>', '')
            
            # 길이 조건 및 알파벳 여부 확인
            if min_len <= len(clean_token) <= max_len and clean_token.isalpha():
                # 실제로 다시 인코딩했을 때 1토큰인지 재검증 (특수 케이스 방지)
                test_encode = tokenizer.encode(clean_token, add_special_tokens=False)
                if len(test_encode) == 1:
                    valid_set.add(clean_token.lower())
        return valid_set

    print(f"Searching for 1-token words between {min_len} and {max_len} characters...")

    set_l = get_valid_tokens(tokenizer_l)
    set_g = get_valid_tokens(tokenizer_g)

    # 2. 두 모델 모두에서 1토큰인 단어(교집합) 찾기
    common_candidates = list(set_l.intersection(set_g))
    common_candidates.sort()

    print(f"Found {len(common_candidates)} potential trigger words.")
    return common_candidates


candidates = find_dual_one_token_candidates(tokenizer_l, tokenizer_g)
print("Preview:", candidates[:10])


sdxl_tokenizers = {'L': tokenizer_l, 'G': tokenizer_g}
sdxl_encoders = {'L': text_encoder1, 'G': text_encoder2}

final_analysis = analyze_sdxl_trigger(candidates, sdxl_tokenizers, sdxl_encoders)
print(final_analysis)
print(f"{'word':10s} | {'sim_L':8s} | {'sim_G':8s} | {'total_score':8s}")
print("-" * 45)
for analysis in final_analysis[:30]:
    word, details, total_score = analysis['word'], analysis['details'], analysis['total_score']
    print(f"{word:10s} | {details['L']['max_sim']:.4f}   | {details['G']['max_sim']:.4f}   | {total_score:.4f}")