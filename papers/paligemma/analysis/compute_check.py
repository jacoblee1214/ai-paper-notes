"""PaliGemma의 시각 토큰 수, 파라미터 수, 학습 토큰·FLOPs·시간 보고값을 산술로 다시 계산한다 (numpy 불필요, 수 초).

구조 값(폭, 깊이 등)은 공식 코드 big_vision의 config에서 읽은 것이다.
  Gemma-2B : width 2048, depth 18, mlp_dim 16384, heads 8, kv heads 1, head_dim 256, vocab 256000(+1024 loc +128 seg)
             (models/ppp/gemma.py get_config, configs/.../transfers/*.py 의 vocab_size=256_000+1024+128)
  So400m/14: width 1152, depth 27, mlp_dim 4304, heads 16, patch 14 (models/vit.py decode_variant), pool_type='none'
  연결기   : 선형 1152→2048 (+bias), 영초기화 (models/vit.py head_zeroinit=True)
TPU v5e 첨두 성능 197 TFLOP/s(bf16)는 공식 문서의 값으로 기억하는 외부 지식이며 이 저장소에서 확인하지 않았다.
출력: compute_results.json
"""
import json
from pathlib import Path

import paper_tables as T

HERE = Path(__file__).resolve().parent
res = {}

# ---------------------------------------------------------------- 시각 토큰
P = 14
res["visual_tokens"] = {str(r): (r // P) ** 2 for r in (224, 448, 896)}
res["visual_tokens_paper"] = {"224": 256, "448": 1024, "896": 4096}
assert res["visual_tokens"] == {k: v for k, v in res["visual_tokens_paper"].items()}
res["video_16_frames_224"] = 16 * 256  # 4096, 논문 §3.2.4

# ---------------------------------------------------------------- 파라미터
def gemma_params(width=2048, depth=18, mlp=16384, heads=8, kv=1, hd=256, vocab=256_000 + 1024 + 128):
    emb = vocab * width                                   # 입력 임베딩 = 출력 가중치(공유)
    attn = width * heads * hd + 2 * width * kv * hd + heads * hd * width
    ffn = 3 * width * mlp                                 # gated(GeGLU): gate, up, down
    norms = 2 * width
    layer = attn + ffn + norms
    return {"embedding": emb, "per_layer": layer, "layers": depth * layer, "final_norm": width,
            "total": emb + depth * layer + width, "non_embedding": depth * layer + width}


def vit_params(width=1152, depth=27, mlp=4304, patch=14, res=224, channels=3):
    pe = patch * patch * channels * width + width
    pos = (res // patch) ** 2 * width
    attn = 4 * (width * width + width)                    # q, k, v, out (+bias)
    ffn = width * mlp + mlp + mlp * width + width
    ln = 2 * 2 * width
    layer = attn + ffn + ln
    return {"patch_embed": pe, "pos_emb": pos, "per_layer": layer, "layers": depth * layer,
            "encoder_norm": 2 * width, "total": pe + pos + depth * layer + 2 * width, "non_pos": pe + depth * layer + 2 * width}


g = gemma_params()
v = {r: vit_params(res=r) for r in (224, 448, 896)}
proj = 1152 * 2048 + 2048
res["params"] = {
    "gemma2b": g, "siglip_so400m_by_res": {str(r): x["total"] for r, x in v.items()}, "linear_projection": proj,
    "total_by_res": {str(r): g["total"] + v[r]["total"] + proj for r in v},
    "paper_claims": "SigLIP-So400m ~400M + Gemma-2B + 선형 투영, 3B 미만(§1, 제목 3B)",
}
tot224 = res["params"]["total_by_res"]["224"]
res["params"]["total_224_B"] = round(tot224 / 1e9, 3)
res["params"]["float32_GiB"] = round(tot224 * 4 / 2 ** 30, 2)
res["params"]["float32_GB"] = round(tot224 * 4 / 1e9, 2)
res["params"]["code_readme_float32_checkpoint"] = "11GB (configs/proj/paligemma/README.md)"
res["params"]["gemma_2b_published_B"] = round(g["total"] / 1e9, 3)  # 공개 Gemma-2B는 2.51B(어휘 256000)
g_orig = gemma_params(vocab=256_000)
res["params"]["gemma2b_vocab256000_B"] = round(g_orig["total"] / 1e9, 3)

# ---------------------------------------------------------------- 학습 FLOPs (예제당)
def flops_per_example(res_px, n_txt, full=False):
    n_img = (res_px // P) ** 2
    seq = n_img + n_txt
    gm = g["non_embedding"]          # 입력 임베딩은 조회라 FLOPs 없음
    logits = g["embedding"]          # 출력 행렬곱은 텍스트 위치에만(코드: text_pre_logits = pre_logits[:, n_img:])
    vit = v[res_px]["non_pos"]
    lin = 2 * (gm * seq + logits * n_txt + vit * n_img + proj * n_img)         # 순전파 행렬곱
    # 전역 어텐션: QK^T와 AV, 층당 4·S²·(heads·head_dim); 마스크가 일부를 가려도 밀집으로 계산된다고 본다(상한)
    attn_gm = 4 * seq * seq * (8 * 256) * 18
    attn_vit = 4 * n_img * n_img * 1152 * 27
    fwd = lin + attn_gm + attn_vit
    return {"n_img": n_img, "seq": seq, "fwd_linear": lin, "fwd_attn_gemma": attn_gm, "fwd_attn_vit": attn_vit,
            "train_6x": 3 * fwd, "train_8x_with_remat": 4 * fwd, "attn_share_of_fwd": round((attn_gm + attn_vit) / fwd, 3)}


s1 = T.STAGE1
st = {"stage1_224": flops_per_example(224, s1["n_txt"])}
for s in T.STAGE2:
    st[f"stage2_{s['res']}"] = flops_per_example(s["res"], s["n_txt"])
res["flops_per_example"] = st

# ---------------------------------------------------------------- 토큰 수
tok = {}
pad_tok_s1 = s1["examples"] * (s1["n_img"] + s1["n_txt"])
tok["stage1_padded_tokens"] = pad_tok_s1
tok["stage1_reported"] = s1["reported_tokens"]
tok["stage1_reported_over_padded"] = round(s1["reported_tokens"] / pad_tok_s1, 3)
tok["stage1_implied_avg_text_tokens_if_img_always_counted"] = round(s1["reported_tokens"] / s1["examples"] - s1["n_img"], 1)
img_only = sum(s["examples"] * s["n_img"] for s in T.STAGE2)
pad_s2 = sum(s["examples"] * (s["n_img"] + s["n_txt"]) for s in T.STAGE2)
tok["stage2_image_tokens_only"] = img_only
tok["stage2_padded_tokens"] = pad_s2
tok["stage2_reported"] = T.STAGE2_REPORTED_TOKENS
tok["stage2_reported_below_image_tokens_alone"] = T.STAGE2_REPORTED_TOKENS < img_only
tok["stage2_per_stage_padded"] = {str(s["res"]): s["examples"] * (s["n_img"] + s["n_txt"]) for s in T.STAGE2}
tok["stage2_per_stage_image_only"] = {str(s["res"]): s["examples"] * s["n_img"] for s in T.STAGE2}
res["tokens"] = tok

# ---------------------------------------------------------------- 시간·처리량
dev, tps = s1["devices"], s1["tok_per_s_per_dev"]
thr = {}
thr["tokens_per_s_total"] = dev * tps
thr["days_for_reported_350B"] = round(s1["reported_tokens"] / (dev * tps) / 86400, 3)
thr["days_for_padded_384B"] = round(pad_tok_s1 / (dev * tps) / 86400, 3)
thr["stage2_hours_each_if_5189_tps"] = {str(s["res"]): round(s["examples"] * (s["n_img"] + s["n_txt"]) / (dev * tps) / 3600, 1) for s in T.STAGE2}
thr["stage2_reported_hours_each"] = 15
thr["stage2_tokens_per_s_per_dev_implied_padded"] = {str(s["res"]): round(s["examples"] * (s["n_img"] + s["n_txt"]) / (dev * 15 * 3600)) for s in T.STAGE2}
thr["stage2_tokens_per_s_per_dev_implied_reported90B"] = round(T.STAGE2_REPORTED_TOKENS / (dev * 2 * 15 * 3600))
# 예제 처리량 비
ex_per_h = {"stage1": 1e9 / (72.0)}  # 'slightly less than 3 days'를 72시간으로 놓음
ex_per_h["stage2_448"] = 50e6 / 15
ex_per_h["stage2_896"] = 10e6 / 15
thr["examples_per_hour_millions"] = {k: round(x / 1e6, 2) for k, x in ex_per_h.items()}
thr["throughput_ratio_stage1_over_448"] = round(ex_per_h["stage1"] / ex_per_h["stage2_448"], 2)
thr["throughput_ratio_stage1_over_896"] = round(ex_per_h["stage1"] / ex_per_h["stage2_896"], 2)
f1 = st["stage1_224"]
thr["flops_ratio_448_over_stage1"] = round(st["stage2_448"]["train_6x"] / f1["train_6x"], 2)
thr["flops_ratio_896_over_stage1"] = round(st["stage2_896"]["train_6x"] / f1["train_6x"], 2)
thr["note"] = "예제 처리량 비가 FLOPs 비와 맞으면 보고된 시간(3일, 15시간)이 같은 활용률로 설명된다. Stage1은 '3일 미만'을 72시간으로 놓은 상한 근사."
res["throughput"] = thr

# ---------------------------------------------------------------- MFU
PEAK = 197e12  # TPU v5e bf16 (외부 지식, 미확인)
mfu = {}
pad_per_tok = f1["train_6x"] / (s1["n_img"] + s1["n_txt"])
rep_per_tok = f1["train_6x"] / (s1["reported_tokens"] / s1["examples"])
for name, f_tok in (("6N_padded384", pad_per_tok), ("6N_if_tokens_are_nonpad350", rep_per_tok)):
    mfu[name] = {"flop_per_token": round(f_tok / 1e9, 2), "tflops_per_dev": round(f_tok * tps / 1e12, 1), "mfu": round(f_tok * tps / PEAK, 3)}
f_remat = f1["train_8x_with_remat"] / (s1["n_img"] + s1["n_txt"])
mfu["8N_remat_padded384"] = {"flop_per_token": round(f_remat / 1e9, 2), "tflops_per_dev": round(f_remat * tps / 1e12, 1), "mfu": round(f_remat * tps / PEAK, 3)}
mfu["paper_reported"] = 0.55
mfu["note"] = ("보고된 5189 tokens/s/device와 MFU 55%를 같이 만족하려면 토큰당 약 20.9 GFLOPs가 필요하다. 6N 모델 FLOPs로는 패딩 포함 토큰 기준 약 38%, "
               "비패딩 350 기준 약 42%이고, 재계산(remat) 포함 8N이면 약 51%에 가깝다. 어떤 정의인지 논문은 쓰지 않는다. 첨두값은 외부 지식(미확인).")
res["mfu"] = mfu

# ---------------------------------------------------------------- 총 학습 연산과 칩-시간
tf = s1["examples"] * st["stage1_224"]["train_6x"] + sum(s["examples"] * st[f"stage2_{s['res']}"]["train_6x"] for s in T.STAGE2)
chip_h = 256 * (72 + 15 + 15)
res["total_training"] = {"flops_6N": tf, "flops_6N_e21": round(tf / 1e21, 2), "chip_hours_v5e": chip_h,
                         "implied_utilisation_6N": round(tf / (chip_h * 3600 * PEAK), 3),
                         "note": "Stage1을 72시간으로 놓은 상한 근사. 전이(20분–10시간, TPUv3-32)는 제외."}

# ---------------------------------------------------------------- 어텐션 마스크 밀도 (prefix-LM 대 인과)
def density(n_img, n_pre, n_suf):
    n = n_img + n_pre + n_suf
    blk = n_img + n_pre
    allowed_prefixlm = blk * blk + sum(blk + i + 1 for i in range(n_suf))
    allowed_causal = n * (n + 1) // 2
    return {"n": n, "prefixlm_allowed": allowed_prefixlm, "causal_allowed": allowed_causal,
            "extra_pairs": allowed_prefixlm - allowed_causal, "density_prefixlm": round(allowed_prefixlm / n ** 2, 3),
            "density_causal": round(allowed_causal / n ** 2, 3)}
res["mask_density"] = {"224_prefix20_suffix12_pad_ignored": density(256, 20, 12), "448_prefix20_suffix12": density(1024, 20, 12),
                       "896_prefix20_suffix12": density(4096, 20, 12),
                       "note": "접두·접미 길이는 예시(실제 과제마다 다름). 이미지·접두 블록이 시퀀스를 거의 채워 차이는 블록 안 위쪽 삼각형이다."}

(HERE / "compute_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("visual_tokens", "params", "tokens", "throughput", "mfu", "total_training")}, ensure_ascii=False, indent=1, default=float))
for k, x in st.items():
    print(k, {a: (f"{b:.3e}" if isinstance(b, float) and b > 1e6 else b) for a, b in x.items()})
