"""Transfusion의 파라미터·FLOPs·토큰·이미지 수 산술 검산 (numpy, 수 초).

논문 근거: 표 2(모델 구성), §4.1(데이터, VAE, 패치, 최적화), 각주 13·15, 표 6·7, 표 9, §4.2의 6ND.
입력: figure_values.json (그림 5의 점 x좌표 = log10 FLOPs)
출력: compute_results.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "figure_values.json").read_text())
out = {}

# 1. Llama형 구성 파라미터(표 2): d, 층 수, ffn = 8/3 d를 256의 배수로 올림(Llama 1 규칙), 어휘 32000(Llama 2 토크나이저), 입출력 임베딩 분리
def llama_params(layers, d, vocab=32000, ffn=None):
    ffn = ffn or int(256 * np.ceil(8 / 3 * d / 256))
    per = 4 * d * d + 3 * d * ffn
    return layers * per + 2 * vocab * d, layers * per, 2 * vocab * d, ffn

cfg = {"0.16B": (16, 768), "0.37B": (24, 1024), "0.76B": (24, 1536), "1.4B": (24, 2048), "7B": (32, 4096)}
out["params"] = {}
for k, (L, d) in cfg.items():
    tot, body, emb, ffn = llama_params(L, d)
    out["params"][k] = {"layers": L, "d": d, "ffn": ffn, "total_B": round(tot / 1e9, 4), "body_B": round(body / 1e9, 4), "embed_B": round(emb / 1e9, 4)}

# 2. 그림 5의 x좌표가 6·N·D(D = 0.1, 0.2, ..., 0.5T)와 맞는가
pts = sorted(V["c4_ppl"]["points"]["Transfusion"])
fl = np.array([10 ** p[0] for p in pts])
rows = []
for k in cfg:
    N = out["params"][k]["total_B"] * 1e9
    pred = np.array([6 * N * d * 1e11 for d in range(1, 6)])
    match = [float(min(abs(np.log(f / p)) for f in fl)) for p in pred]
    rows.append({"size": k, "max_abs_log_err_over_5_checkpoints": round(max(match), 4)})
out["fig5_x_vs_6ND_(D=0.1..0.5T)"] = rows
out["fig5_same_x_for_both_methods_(c4,wiki,llama panels)"] = all(
    sorted(round(p[0], 2) for p in V[n]["points"]["Chameleon"]) == sorted(round(p[0], 2) for p in V[n]["points"]["Transfusion"])
    for n in ("c4_ppl", "wiki_ppl", "llama_acc"))

# 3. U-Net 추가 파라미터 (표 7의 ΔEnc/Dec 비율 × 모델 크기, §4.1의 "0.27B across all configurations"와 비교)
t7 = {"0.16B": (0.16, 106.1), "0.37B": (0.37, 71.3), "0.76B": (0.76, 35.5), "1.4B": (1.4, 19.3), "7B": (7.0, 3.8)}
out["unet_extra_params_B_implied_by_table7"] = {k: round(n * p / 100, 3) for k, (n, p) in t7.items()}
out["unet_extra_params_B_if_0.27_const_pct_of_actual_N"] = {k: round(0.27 / out["params"][k]["total_B"] * 100, 1) for k in cfg}
out["table9_7.3B_vs_6.74B+0.27B"] = round(7.3 - (out["params"]["7B"]["total_B"] + 0.27), 3)

# 4. 이미지당 시퀀스 길이와 학습 예산당 본 이미지 수 (0.5T 토큰, 텍스트:이미지 = 1:1 -> 이미지 쪽 0.25T. 캡션 토큰은 무시)
latent = 32 * 32
def patches(p):
    return latent // (p * p)
img_tokens = 0.25e12
n_img = 380e6
tab = {}
for p in (1, 2, 4, 8):
    P = patches(p)
    tab[f"patch{p}x{p}"] = {"patches_per_image": P, "images_seen_B": round(img_tokens / P / 1e9, 3), "epochs_over_380M": round(img_tokens / P / n_img, 1)}
tab["chameleon_VQ_8x8_pixels_per_token"] = {"tokens_per_image": 1024, "images_seen_B": round(img_tokens / 1024 / 1e9, 3), "epochs_over_380M": round(img_tokens / 1024 / n_img, 2)}
out["images_seen_in_0.5T_budget"] = tab
out["images_ratio_transfusion2x2_over_chameleon"] = tab["patch2x2"]["images_seen_B"] / tab["chameleon_VQ_8x8_pixels_per_token"]["images_seen_B"]
# §4.4: 2T 토큰 = 1T 텍스트 + 이미지 쪽
out["sec4.4_image_side"] = {"692M_pairs_x5_epochs": 692e6 * 5, "x256_patches_T": 692e6 * 5 * 256 / 1e12, "paper_says_3.5B_pairs": 3.5e9, "3.5B_x256_T": 3.5e9 * 256 / 1e12}

# 5. 생성 추론 FLOPs: Transfusion 250 diffusion 단계(§4.1) × 이미지당 패치 vs Chameleon 이미지당 토큰 (프롬프트 KV 캐시 가정, 어텐션 제곱항 제외, CFG는 양쪽 2배)
N = out["params"]["7B"]["total_B"] * 1e9
ft = 2 * N * 250 * 256          # 단계마다 전 패치를 다시 순전파
fc = 2 * N * 1024               # 자기회귀: 토큰 한 번씩
out["generation_flops_per_image_7B"] = {
    "transfusion_250steps_256patches": ft, "chameleon_1024tokens": fc, "ratio_transfusion_over_chameleon": ft / fc,
    "note": "CFG는 양쪽 모두 2배라 비율 불변. 논문의 6ND 동등 FLOP 비교는 학습 계산량이고 서비스 비용이 아니다(논문 각주 15는 어텐션 제곱항만 다룸).",
}
# 6. 학습 FLOPs의 어텐션 제곱항 크기: 시퀀스 4096, 7B, 층당 어텐션 12·L·d·T (토큰당) 대비 6N
d, L, T = 4096, 32, 4096
attn_per_tok = 6 * 2 * L * d * T / 2   # 인과 평균 T/2 길이, QK와 AV 두 번, 순+역 3배 중 6 -> 근사
out["attention_term_fraction_7B_T4096"] = attn_per_tok / (6 * N + attn_per_tok)
# 7. 학습 FLOPs 총량
out["train_flops_7B_0.5T_6ND"] = 6 * N * 0.5e12
out["train_flops_7B_2T_6ND_4.4"] = 6 * (N + 0.27e9) * 2e12
(HERE / "compute_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:3500])
