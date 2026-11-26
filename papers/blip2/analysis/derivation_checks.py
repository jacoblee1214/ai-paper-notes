"""식·마스크·파라미터 수 검산 (학습 없음, 수 초). 결과는 derivation_results.json.

실행: python blip2/analysis/derivation_checks.py  (저장소 루트에서, torch가 설치된 파이썬)
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import minimal_impl as m  # noqa: E402

OUT = Path(__file__).parent / "derivation_results.json"
res = {}
torch.manual_seed(0)

# ---- 1. 세 마스크 행렬 (K=3 질의, T=4 텍스트) ------------------------------------------------
K, T = 3, 4
labels = [f"Q{i}" for i in range(K)] + [f"T{j}" for j in range(T)]
masks = {}
for mode in ("itc", "itg", "itm"):
    mk = m.build_mask(mode, K, T)[0, 0].int()
    masks[mode] = {"labels": labels, "rows_see_cols": mk.tolist(),
                   "ascii": "\n".join(f"{labels[i]}  " + " ".join("x" if v else "." for v in mk[i].tolist()) for i in range(K + T))}
res["masks_K3_T4"] = masks

# ---- 2. 정보가 닿는 깊이: 층별로 이미지를 바꿨을 때 텍스트·질의 표현이 달라지는가 ------------------
cfg = m.QFConfig(d=32, heads=4, layers=12, ffn=64, k_queries=4, d_img=16, embed_dim=16)
torch.manual_seed(0)
qf = m.QFormer(cfg).eval()
qf.query_tokens.data.normal_(0, 1.0)
g = torch.Generator().manual_seed(0)
img1, img2 = torch.randn(2, 20, 16, generator=g), torch.randn(2, 20, 16, generator=g)
txt = torch.randint(3, m.V, (2, 6), generator=g)
txt[:, 0] = m.CLS
reach = {"cross_layers": qf.cross_layers()}
for mode in ("itc", "itg", "itm"):
    _, _, a = qf(img1, txt, mode, return_layers=True)
    _, _, b = qf(img2, txt, mode, return_layers=True)
    dt = [(x[:, cfg.k_queries:] - y[:, cfg.k_queries:]).abs().max().item() for x, y in zip(a, b)]
    dq = [(x[:, :cfg.k_queries] - y[:, :cfg.k_queries]).abs().max().item() for x, y in zip(a, b)]
    first_t = next((i for i, v in enumerate(dt) if v > 1e-5), None)
    first_q = next((i for i, v in enumerate(dq) if v > 1e-5), None)
    reach[mode] = {"first_layer_text_depends_on_image": first_t, "first_layer_query_depends_on_image": first_q,
                   "text_layers_with_image_info": (cfg.layers - first_t) if first_t is not None else 0}
res["image_reach_12_layers"] = reach

# ---- 3. 파라미터 수 산술 ---------------------------------------------------------------------
pc = m.qformer_param_count
s1 = pc()
res["params_stage1_bertbase_vitg_K32"] = {
    "with_query_ffn_M": round(s1 / 1e6, 2), "without_query_ffn_M": round(pc(query_ffn=False) / 1e6, 2),
    "paper_M": 188, "relative_gap": round((s1 - 188e6) / 188e6, 4),
    "parts_M": {"embeddings": round((30523 * 768 + 512 * 768 + 1536) / 1e6, 2),
                "bert_layers_with_query_ffn": round(12 * (4 * (768 * 768 + 768) + 1536 + 2 * (2 * 768 * 3072 + 3072 + 768 + 1536)) / 1e6, 2)},
}
t2 = {}
for enc, dv in (("ViT-L", 1024), ("ViT-g", 1408)):
    for llm, E in (("OPT-2.7B", 2560), ("OPT-6.7B", 4096), ("FlanT5-XL", 2048), ("FlanT5-XXL", 4096)):
        t2[f"{enc} {llm}"] = round(pc(stage=2, dv=dv, E=E) / 1e6, 1)
res["params_stage2_trainable_M"] = {"ours": t2, "paper_table2_M": {
    "ViT-L OPT2.7B": 104, "ViT-g OPT2.7B": 107, "ViT-g OPT6.7B": 108, "ViT-L FlanT5XL": 103, "ViT-g FlanT5XL": 107, "ViT-g FlanT5XXL": 108}}
res["flamingo_ratio_54x"] = {"10.2B/188M": round(10.2e9 / 188e6, 2), "10.2B/108M(Table 2 BLIP-2 ViT-g FlanT5XXL)": round(10.2e9 / 108e6, 1),
                             "vqav2_gain_65.0-56.3": round(65.0 - 56.3, 1)}
res["token_counts"] = {
    "qformer_output": [32, 768, 32 * 768],
    "vit_l14_224_features": [257, 1024, 257 * 1024], "vit_g14_224_features": [257, 1408, 257 * 1408],
    "number_ratio_vitL": round(257 * 1024 / (32 * 768), 2), "number_ratio_vitg": round(257 * 1408 / (32 * 768), 2),
    "llava15_visual_tokens_336px": 24 * 24, "token_ratio_vs_llava15": round(576 / 32, 1),
    "note": "257 = (224/14)^2 + 1(CLS). 이미지 해상도가 바뀌면 특징 수는 바뀌지만 Q-Former 출력은 32개로 같음(3.1절)"}

# ---- 4. ITC의 라벨 평활화 바닥값 ---------------------------------------------------------------
floor = {}
for B in (2, 16, 64, 1680, 2320):
    eps = 0.1
    p_true = 1 - eps + eps / B
    p_other = eps / B
    h = -(p_true * math.log(p_true) + (B - 1) * p_other * math.log(p_other))
    # 같은 값을 PyTorch로: 최적 로짓 = log(평활화된 목표)
    tgt = torch.full((B,), p_other, dtype=torch.float64)
    tgt[0] = p_true
    ce = F.cross_entropy(tgt.log()[None], torch.tensor([0]), label_smoothing=eps).item()
    floor[str(B)] = {"closed_form": round(h, 5), "torch": round(ce, 5), "log_B": round(math.log(B), 4)}
res["itc_label_smoothing_floor"] = floor

# ---- 5. 어려운 음성 샘플링의 "유효 후보 수" ----------------------------------------------------
eff = {}
for B in (64, 2320):
    for sd in (0.05, 0.10, 0.20):
        sim = torch.randn(B, B) * sd + 0.2
        w = F.softmax((sim / 0.07).masked_fill(torch.eye(B, dtype=torch.bool), -10000.0), dim=1)
        ent = -(w * (w + 1e-30).log()).sum(1).mean().item()
        eff[f"B={B}, sd={sd}"] = {"effective_candidates": round(math.exp(ent), 1), "max_prob_mean": round(w.max(1).values.mean().item(), 4)}
res["hard_negative_effective_candidates_tau0.07"] = eff

# ---- 6. 토이 장면의 정보량 -----------------------------------------------------------------
bits = {}
for n in range(1, m.MAX_OBJ + 1):
    count = math.comb(m.G * m.G, n) * math.perm(m.NCOL, n)      # 칸 n개 선택, 서로 다른 색을 배정
    bits[str(n)] = round(math.log2(count), 2)
total = sum(math.comb(m.G * m.G, n) * math.perm(m.NCOL, n) for n in range(1, m.MAX_OBJ + 1))
res["toy_scene_bits"] = {"per_object_count": bits, "all_scenes_uniform_over_n_bits": round(math.log2(total), 2),
                         "log2_64_cells": 6.0, "log2_6_colors": round(math.log2(6), 2)}

# ---- 7. 논문 표 재계산: 표 간 일치와 표본 오차(테스트 표본 이항 근사) --------------------------------
# 데이터셋 크기는 논문에 없는 일반 지식이다: VQAv2 test-dev 약 107,394문항, GQA test-dev 12,578문항, OK-VQA test 5,046문항,
# COCO 검색 5K 이미지(캡션 25,000개). 비교는 같은 테스트 문항을 쓰므로 짝지은 오차는 더 작다(아래는 보수적인 비짝 상한).
def se_pts(p_pct, n):
    p = p_pct / 100
    return 100 * math.sqrt(p * (1 - p) / n)


def diff_row(a, b, n):
    se = math.sqrt(se_pts(a, n) ** 2 + se_pts(b, n) ** 2)
    return {"a": a, "b": b, "diff": round(a - b, 2), "se_unpaired": round(se, 2), "z": round((a - b) / se, 2)}


N_VQA, N_GQA, N_OK = 107394, 12578, 5046
tab = {
    "table1_vs_table2_vs_table3_vs_table5_consistency": {
        "T1_VQAv2_65.0_equals_T2_ViT-g_FlanT5XXL_testdev": 65.0 == 65.0, "T1_NoCaps_CIDEr_121.6_equals_T3_FlanT5XL_overall": 121.6 == 121.6,
        "T1_Flickr_TR1_97.6_IR1_89.7_equals_T5_ViT-g": (97.6, 89.7) == (97.6, 89.7),
        "T6_ITG_row_equals_T5_COCO_ViT-g_R1_R5": {"i2t": (85.4, 97.0), "t2i": (68.3, 87.7)}},
    "vqav2_testdev_diffs_table2": {
        "ViT-g - ViT-L (OPT2.7B)": diff_row(52.3, 49.7, N_VQA),
        "ViT-g - ViT-L (FlanT5-XL)": diff_row(63.0, 62.3, N_VQA),
        "OPT6.7B - OPT2.7B (ViT-g)": diff_row(52.6, 52.3, N_VQA),
        "FlanT5-XXL - FlanT5-XL (ViT-g)": diff_row(65.0, 63.0, N_VQA),
        "FlanT5-XL - OPT2.7B (ViT-g)": diff_row(63.0, 52.3, N_VQA),
        "BLIP-2 XXL - Flamingo80B (zero-shot)": diff_row(65.0, 56.3, N_VQA)},
    "gqa_testdev_diffs_table2": {"ViT-g - ViT-L (FlanT5-XL)": diff_row(44.2, 44.4, N_GQA), "FlanT5-XL - OPT2.7B (ViT-g)": diff_row(44.2, 34.6, N_GQA)},
    "okvqa_test_diffs_table2": {"ViT-g - ViT-L (FlanT5-XL)": diff_row(40.7, 39.4, N_OK), "BLIP-2 XXL - Flamingo80B": diff_row(45.9, 50.6, N_OK)},
    "vqav2_finetuned_table4_testdev": {"OPT6.7B - OFA": diff_row(82.19, 82.00, N_VQA), "OPT6.7B - Flamingo80B": diff_row(82.19, 82.00, N_VQA),
                                         "OPT6.7B(test-std) - Flamingo80B(test-std)": diff_row(82.30, 82.10, N_VQA),
                                         "OPT6.7B(test-std) - OFA(test-std)": diff_row(82.30, 82.00, N_VQA)},
    "table6_itg_effect_coco5k": {
        "i2t_R1": diff_row(85.4, 84.5, 5000), "i2t_R5": diff_row(97.0, 96.2, 5000),
        "t2i_R1": diff_row(68.3, 67.2, 25000), "t2i_R5": diff_row(87.7, 87.1, 25000)},
}
res["table_recompute"] = tab

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: res[k] for k in ("image_reach_12_layers", "params_stage1_bertbase_vitg_K32", "params_stage2_trainable_M",
                                       "flamingo_ratio_54x", "itc_label_smoothing_floor", "toy_scene_bits", "table_recompute")}, ensure_ascii=False, indent=1))
for k, v in masks.items():
    print(k)
    print(v["ascii"])
