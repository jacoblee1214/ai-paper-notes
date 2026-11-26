"""Prismatic VLMs(arXiv:2402.07865) 부록 표 2-4 재계산: 설계 축별 비교의 표본 오차, 부호 검정, 검정 방법 민감도, 계산량 산술.

입력: analysis/tables.json (parse_tables.py가 논문 부록 표 2-4에서 만듦)
출력: analysis/recompute_results.json, 표준 출력에 요약
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(결과 해석에 영향):
- 벤치마크 문항 수는 공식 평가 코드(vlm_eval/conf/datasets.py)의 expected_examples. RefCOCO 계열은 합 26488만 코드에 있고
  개별 크기(10834, 10758, 4896)는 표준 val 분할 크기(합이 26488과 일치).
- VQAv2, VizWiz, TextVQA는 soft accuracy라 이항 구간은 근사이다. 같은 이미지의 여러 문항(군집)은 무시한다(구간이 낙관적).
- 같은 문항에 대한 두 모델의 불일치율 q는 공개되지 않았다. q = 0.1, 0.2, 0.3을 가정하고 MDE ~ 2.80 sqrt(q/n)를 낸다.
"""
import json
import math
import unicodedata
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
T = json.loads((HERE / "tables.json").read_text(encoding="utf-8"))
NFC = lambda s: unicodedata.normalize("NFC", s)

B12 = ["VQAv2", "GQA", "VizWiz", "TextVQA", "RefCOCO", "RefCOCO+", "RefCOCOg", "OCIDRef", "VSR", "POPE", "TallyQA", "AI2D"]
N_Q = {"VQAv2": 214354, "GQA": 12578, "VizWiz": 4319, "TextVQA": 5000, "RefCOCO": 10834, "RefCOCO+": 10758,
       "RefCOCOg": 4896, "OCIDRef": 18342, "VSR": 1222, "POPE": 9000, "TallyQA": 38589, "AI2D": 15501}

# ---------- 표 읽기 ----------
ROWS = {}
for i in range(len(T["T2"]["rows"])):
    sc = {}
    for t in ("T2", "T3", "T4"):
        r = T[t]["rows"][i]
        assert r["section"] == T["T2"]["rows"][i]["section"]  # 표마다 모델 이름이 조금 다르다(T2 이름을 키로 씀)
        sc.update(r["scores"])
    ROWS[(NFC(T["T2"]["rows"][i]["section"]), NFC(T["T2"]["rows"][i]["model"]))] = sc


def R(sec, name):
    return ROWS[(NFC(sec), NFC(name))]


def vec(m):
    return np.array([np.nan if m[b] is None else m[b] for b in B12], float)


RP, OPT, VR, IP, EN = ("Reproduction & Optimization Procedure", "Official Models", "Visual Representation",
                       "Image Preprocessing", "Ensembling Visual Features")
LM, BL, CT, ST, SD, PR7, PR13 = ("Base vs. Instruct Tuned LMs", "Better LMs", "Co-training on Language Safety Data",
                                 "Scaling Train Time", "Scaling Data", "Prism 7B", "Prism 13B")

out = {}

# ---------- 0. 표 무결성: 같은 모델이 여러 곳에 반복되는 행 ----------
ident = [
    ("7B 기본(CLIP 336 letterbox, Vicuna, 1 epoch)", [(RP, "Single-Stage 7B"), (RP, "Frozen ViT (Single-Stage)"), (IP, "CLIP ViT-L 336px (Letterbox)"),
                                                    (LM, "Vicuña v1.5 7B"), (CT, "Vicuña v1.5 7B"), (ST, "1 Epoch"), (SD, "Base")]),
    ("13B 기본", [(RP, "Single-Stage 13B"), (LM, "Vicuña v1.5 13B")]),
    ("Llama-2 7B", [(LM, "Llama-2 7B"), (CT, "Llama-2 7B")]),
    ("CLIP 336 naive", [(IP, "CLIP ViT-L 336px (Naive Resize)"), (EN, "CLIP 336px (Naive Resize)")]),
    ("SigLIP 384 naive", [(IP, "SigLIP ViT-SO 384px (Naive Resize)"), (EN, "SigLIP 384px (Naive Resize)")]),
]
out["identity_rows"] = {}
for label, keys in ident:
    ref = vec(R(*keys[0]))
    same = all(np.array_equal(ref, vec(R(*k)), equal_nan=True) for k in keys[1:])
    out["identity_rows"][label] = {"rows": len(keys), "all_equal": bool(same)}

# VSR 다수 클래스: 629/1222 = 0.5147
vsr_all = {k: v["VSR"] for k, v in ROWS.items()}
uniq_models = {}
for (s, n), v in ROWS.items():
    uniq_models.setdefault(tuple(vec(v)), (s, n))
n_models = len(uniq_models)
n_collapse = sum(1 for k in uniq_models if abs(k[B12.index("VSR")] - 51.47) < 0.005)
n_near = sum(1 for k in uniq_models if k[B12.index("VSR")] < 55.0)
out["vsr"] = {"always_true_acc_629_of_1222": round(629 / 1222 * 100, 2), "unique_models": n_models,
              "models_exactly_51.47": n_collapse, "models_below_55": n_near,
              "wilson95_of_baseline": [round(100 * x, 2) for x in
                                       (lambda p, n, z=1.96: ((p + z * z / 2 / n - z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n)) / (1 + z * z / n),
                                                              (p + z * z / 2 / n + z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n)) / (1 + z * z / n)))(0.5147, 1222)]}

# ---------- 1. 문항 수, Wilson 구간, MDE ----------
def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / 2 / n) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / d
    return c - h, c + h


def mde_pts(n, q):  # 점(%p)
    return 100 * 2.80 * math.sqrt(q / n)


base = vec(R(IP, "CLIP ViT-L 336px (Letterbox)"))
bench = {}
for b, x in zip(B12, base):
    n = N_Q[b]
    lo, hi = wilson(x / 100, n)
    bench[b] = {"n": n, "base_acc": x, "wilson95_halfwidth_pts": round(50 * (hi - lo), 2),
                "mde_pts": {f"q={q}": round(mde_pts(n, q), 2) for q in (0.1, 0.2, 0.3)}}
out["benchmarks"] = bench
out["refcoco_sum_check"] = {"sum": 10834 + 10758 + 4896, "code_expected_examples": 26488}

# ---------- 2. z 점수 ----------
pool = np.array([k for k in uniq_models])  # 고유 행
MU, SDV = np.nanmean(pool, 0), np.nanstd(pool, 0)  # 모집단 표준편차(공식 compute_z_scores.py와 같은 np.std)
allrows = np.array([vec(v) for v in ROWS.values()])
MU2, SD2 = np.nanmean(allrows, 0), np.nanstd(allrows, 0)
out["z_pool"] = {"unique_rows": len(pool), "all_rows": len(allrows),
                 "bench_sd_pts": {b: round(float(s), 2) for b, s in zip(B12, SDV)}}


def z(m, mu=MU, sd=SDV):
    return (vec(m) - mu) / sd


# ---------- 3. 설계 축별 비교 ----------
# 각 실험: 이름 -> (논문 p, 방향 설명, [(라벨, 기준 행, 대안 행), ...])
EXP = {
    "단일 단계 (그림 4)": (0.00558, "다단계 -> 단일 단계", [
        ("7B", R(RP, "LLaVa v1.5 7B (Reproduction)"), R(RP, "Single-Stage 7B")),
        ("13B", R(RP, "LLaVa v1.5 13B (Reproduction)"), R(RP, "Single-Stage 13B"))]),
    "시각 인코더 미동결 (그림 5)": (0.00381, "동결 -> 미세조정 (낮을 것)", [
        ("단일 단계", R(RP, "Frozen ViT (Single-Stage)"), R(RP, "Finetune ViT (Single-Stage)")),
        ("다단계", R(RP, "LLaVa v1.5 7B (Reproduction)"), R(RP, "Finetune ViT (Multi-Stage)"))]),
    "인코더 (그림 6 왼쪽)": (7.11e-8, "IN1K/DINOv2 -> CLIP/SigLIP (224px)", [
        ("IN1K->CLIP", R(VR, "IN1K ViT-L 224px"), R(VR, "CLIP ViT-L 224px")),
        ("IN1K->SigLIP", R(VR, "IN1K ViT-L 224px"), R(VR, "SigLIP ViT-SO 224px")),
        ("DINOv2->CLIP", R(VR, "DINOv2 ViT-L 224px"), R(VR, "CLIP ViT-L 224px")),
        ("DINOv2->SigLIP", R(VR, "DINOv2 ViT-L 224px"), R(VR, "SigLIP ViT-SO 224px"))]),
    "전처리 letterbox->naive (그림 6 가운데)": (0.0176, "letterbox -> naive resize", [
        ("CLIP 336", R(IP, "CLIP ViT-L 336px (Letterbox)"), R(IP, "CLIP ViT-L 336px (Naive Resize)")),
        ("SigLIP 384", R(IP, "SigLIP ViT-SO 384px (Letterbox)"), R(IP, "SigLIP ViT-SO 384px (Naive Resize)"))]),
    "전처리 letterbox->naive (융합 포함, 4쌍)": (None, "논문은 CLIP, SigLIP만 검정", [
        ("CLIP 336", R(IP, "CLIP ViT-L 336px (Letterbox)"), R(IP, "CLIP ViT-L 336px (Naive Resize)")),
        ("SigLIP 384", R(IP, "SigLIP ViT-SO 384px (Letterbox)"), R(IP, "SigLIP ViT-SO 384px (Naive Resize)")),
        ("DINO+CLIP 336", R(EN, "DINOv2 + CLIP 336px (Letterbox)"), R(EN, "DINOv2 + CLIP 336px (Naive Resize)")),
        ("DINO+SigLIP 384", R(EN, "DINOv2 + SigLIP 384px (Letterbox)"), R(EN, "DINOv2 + SigLIP 384px (Naive Resize)"))]),
    "전처리 letterbox->resize&crop": (None, "크롭이 나쁘다는 논문 서술", [
        ("CLIP 336", R(IP, "CLIP ViT-L 336px (Letterbox)"), R(IP, "CLIP ViT-L 336px (Resize Crop)")),
        ("SigLIP 384", R(IP, "SigLIP ViT-SO 384px (Letterbox)"), R(IP, "SigLIP ViT-SO 384px (Resize Crop)"))]),
    "해상도 (그림 6 오른쪽)": (6.05e-4, "224px -> 336/384px (letterbox)", [
        ("CLIP", R(VR, "CLIP ViT-L 224px"), R(IP, "CLIP ViT-L 336px (Letterbox)")),
        ("SigLIP", R(VR, "SigLIP ViT-SO 224px"), R(IP, "SigLIP ViT-SO 384px (Letterbox)"))]),
    "융합 DINOv2+SigLIP (그림 7 왼쪽)": (0.00164, "SigLIP -> DINOv2+SigLIP", [
        ("naive", R(IP, "SigLIP ViT-SO 384px (Naive Resize)"), R(EN, "DINOv2 + SigLIP 384px (Naive Resize)")),
        ("letterbox", R(IP, "SigLIP ViT-SO 384px (Letterbox)"), R(EN, "DINOv2 + SigLIP 384px (Letterbox)"))]),
    "융합 DINOv2+CLIP (그림 7 왼쪽)": (0.37313, "CLIP -> DINOv2+CLIP", [
        ("naive", R(IP, "CLIP ViT-L 336px (Naive Resize)"), R(EN, "DINOv2 + CLIP 336px (Naive Resize)")),
        ("letterbox", R(IP, "CLIP ViT-L 336px (Letterbox)"), R(EN, "DINOv2 + CLIP 336px (Letterbox)"))]),
    "LM 지시 튜닝 -> base (그림 7 오른쪽)": (0.34854, "Vicuna(instruct) -> Llama-2(base)", [
        ("7B", R(LM, "Vicuña v1.5 7B"), R(LM, "Llama-2 7B")),
        ("13B", R(LM, "Vicuña v1.5 13B"), R(LM, "Llama-2 13B"))]),
    "더 좋은 LM (그림 12)": (0.03097, "Llama-2/Vicuna -> Mistral", [
        ("base", R(LM, "Llama-2 7B"), R(BL, "Mistral v1 7B")),
        ("instruct", R(LM, "Vicuña v1.5 7B"), R(BL, "Mistral Instruct v1 7B"))]),
    "언어 전용 공동 학습 제거 (그림 8)": (0.13655, "공동 학습 -> 제거", [
        ("Vicuna", R(CT, "Vicuña v1.5 7B"), R(CT, "Vicuña v1.5 7B (No Co-training)")),
        ("Llama-2", R(CT, "Llama-2 7B"), R(CT, "Llama-2 7B (No Co-training)"))]),
    "학습 시간 1->2 epoch (그림 10)": (0.00496, "1 epoch -> 2 epochs", [("7B", R(ST, "1 Epoch"), R(ST, "2 Epochs"))]),
    "학습 시간 2->3 epoch": (None, "논문은 2 epoch에서 포화라고 씀", [("7B", R(ST, "2 Epochs"), R(ST, "3 Epochs"))]),
    "데이터 추가 (그림 10)": (0.01459, "기본 -> +LVIS-4V+LRV", [("7B", R(SD, "Base"), R(SD, "Base + LVIS-4V + LRV"))]),
}


def sg(x):  # 유효숫자 3자리
    return float(f"{x:.3g}")


def sign_test(d):
    d = d[~np.isnan(d)]
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    if pos + neg == 0:
        return pos, neg, 1.0
    return pos, neg, float(stats.binomtest(pos, pos + neg, 0.5).pvalue)


res = {}
for name, (p_paper, desc, pairs) in EXP.items():
    zb = np.array([z(b) for _, b, a in pairs])
    za = np.array([z(a) for _, b, a in pairs])
    zb2 = np.array([z(b, MU2, SD2) for _, b, a in pairs])
    za2 = np.array([z(a, MU2, SD2) for _, b, a in pairs])
    dz = (za - zb)
    flat = dz[~np.isnan(dz)]
    raw = np.array([vec(a) - vec(b) for _, b, a in pairs])
    pos, neg, p_sign = sign_test(raw.ravel())
    per_pair = []
    for (lab, b, a), r, d in zip(pairs, raw, dz):
        pp, nn, ps = sign_test(r)
        per_pair.append({"pair": lab, "delta_pts": {k: round(float(v), 2) for k, v in zip(B12, r)}, "pos": pp, "neg": nn,
                         "sign_p_two_sided": sg(ps), "mean_z_diff": round(float(np.nanmean(d)), 3),
                         "n_exceed_mde_q0.2": int(sum(abs(v) > mde_pts(N_Q[k], 0.2) for k, v in zip(B12, r) if not np.isnan(v)))})
    # 벤치마크를 하나씩 빼도 평균 z 차이의 부호가 같은가
    mean_dz = np.nanmean(dz, 0)
    loo = [float(np.nanmean(np.delete(mean_dz, j))) for j in range(len(B12))]
    row = {"desc": desc, "n_pairs": len(pairs), "paper_p": p_paper,
           "mean_z_diff": round(float(np.nanmean(dz)), 3),
           "pos": pos, "neg": neg, "sign_p_two_sided_flat": sg(p_sign),
           "loo_mean_z_sign_stable": bool(all(np.sign(x) == np.sign(np.nanmean(dz)) for x in loo)),
           "mean_z_diff_without_VSR": round(float(np.nanmean(np.delete(dz, B12.index("VSR"), 1))), 3),
           "p_alt_gt_flat_t": sg(float(stats.ttest_1samp(flat, 0, alternative="greater").pvalue)),
           "p_alt_lt_flat_t": sg(float(stats.ttest_1samp(flat, 0, alternative="less").pvalue)),
           "p_alt_gt_welch": sg(float(stats.ttest_ind(za[~np.isnan(za)], zb[~np.isnan(zb)], equal_var=False, alternative="greater").pvalue)),
           "p_alt_gt_flat_t_duplicates_in_pool": sg(float(stats.ttest_1samp((za2 - zb2).ravel(), 0, alternative="greater").pvalue)),
           "per_pair": per_pair}
    try:
        row["p_alt_gt_wilcoxon"] = sg(float(stats.wilcoxon(flat, alternative="greater").pvalue))
    except ValueError:
        row["p_alt_gt_wilcoxon"] = None
    # 쌍 단위(벤치마크를 평균한 뒤) 검정: 쌍이 2개 이상일 때만
    pair_means = np.nanmean(dz, 1)
    row["pair_level_mean_z"] = [round(float(x), 3) for x in pair_means]
    if len(pair_means) >= 3:
        row["p_alt_gt_pair_level_t"] = sg(float(stats.ttest_1samp(pair_means, 0, alternative="greater").pvalue))
    res[name] = row
out["axes"] = res

# 인코더 비교의 교란: 패치 크기와 토큰 수
# ---------- 4. 노이즈 하한: 공식 LLaVA 대 재현 ----------
nf = {}
for sc, o, r in (("7B", R(OPT, "LLaVa v1.5 7B"), R(RP, "LLaVa v1.5 7B (Reproduction)")),
                 ("13B", R(OPT, "LLaVa v1.5 13B"), R(RP, "LLaVa v1.5 13B (Reproduction)"))):
    d = vec(r) - vec(o)
    se = np.array([100 * math.sqrt(0.2 / N_Q[b]) for b in B12])
    nf[sc] = {"delta_pts": {b: round(float(x), 2) for b, x in zip(B12, d)},
              "delta_in_se_q0.2": {b: round(float(x / s), 1) for b, x, s in zip(B12, d, se)},
              "rms_pts": round(float(np.sqrt(np.mean(d ** 2))), 2),
              "n_beyond_mde_q0.2": int(sum(abs(x) > 2.8 * s for x, s in zip(d, se))),
              "mean_z_diff": round(float(np.mean(d / SDV)), 3), "sd_z_diff": round(float(np.std(d / SDV, ddof=1)), 3),
              "abs_mean_z_diff_by_bench": {b: round(float(abs(x) / s), 2) for b, x, s in zip(B12, d, SDV)}}
out["noise_floor_official_vs_repro"] = nf

# ---------- 5. 몬테카를로: 공유 학습 노이즈가 있을 때 논문식 검정의 거짓 양성률 ----------
rng = np.random.default_rng(0)
sig_ind = np.array([100 * math.sqrt(0.2 / N_Q[b]) / SDV[i] for i, b in enumerate(B12)])  # z 단위, 쌍 차이의 평가 표본 노이즈
mc = {}
for k in (1, 2, 4):
    for tau in (0.0, 0.05, 0.10, 0.20, 0.30):  # 쌍 차이의 공유 성분 표준편차(z 단위)
        fp = 0
        n_sim = 20000
        for _ in range(n_sim):
            shared = rng.normal(0, tau, size=(k, 1))
            d = shared + rng.normal(0, 1, size=(k, 12)) * sig_ind
            if stats.ttest_1samp(d.ravel(), 0, alternative="greater").pvalue < 0.01:
                fp += 1
        mc[f"k={k},tau={tau}"] = round(fp / n_sim, 4)
out["mc_false_positive_rate_alpha0.01"] = mc
out["mc_sig_ind_z"] = {b: round(float(s), 4) for b, s in zip(B12, sig_ind)}

# ---------- 6. 계산량 산술 ----------
def patches(res, patch=14):
    return (res // patch) ** 2


tok = {"CLIP-L/14 224": patches(224), "CLIP-L/14 336": patches(336), "SigLIP-SO/14 224": patches(224), "SigLIP-SO/14 384": patches(384),
       "IN1K ViT-L/16 224": patches(224, 16), "fused 224": patches(224), "fused 384": patches(384)}


def mlp2(dv, dl):
    return dv * dl + dl + dl * dl + dl


def mlp3(dv, dl):
    h = 4 * dv
    return dv * h + h + h * dl + dl + dl * dl + dl


def vit_params(width, depth, mlp, patch=14, ch=3):
    per = 4 * width * width + 4 * width + 2 * width * mlp + mlp + width + 4 * width  # qkv+proj, mlp, 2 LN
    return depth * per + patch * patch * ch * width + width


vit_l = vit_params(1024, 24, 4096)
vit_so = vit_params(1152, 27, 4304)
proj = {}
for lm, dl in (("Llama-2 7B", 4096), ("Llama-2 13B", 5120)):
    proj[lm] = {"CLIP-L (1024) 2층 MLP": mlp2(1024, dl), "SigLIP-SO (1152) 2층 MLP": mlp2(1152, dl),
                "DINOv2+SigLIP (2176) 논문 서술대로 2층 MLP": mlp2(2176, dl),
                "DINOv2+SigLIP (2176) 코드의 FusedMLPProjector 3층": mlp3(2176, dl),
                "DINOv2+CLIP (2048) 코드 3층": mlp3(2048, dl)}
out["tokens"] = tok
out["projector_params"] = proj
out["vit_params_analytic"] = {"ViT-L/14": vit_l, "ViT-SO400M/14": vit_so}

LLM = {"7B": dict(N=6.74e9, d=4096, L=32), "13B": dict(N=13.02e9, d=5120, L=40)}


def llm_fwd_flops(seq, m):  # 2N 토큰당 + 어텐션 4 L d s^2 (인과 마스크 반을 안 쳐서 상한)
    return 2 * m["N"] * seq + 4 * m["L"] * m["d"] * seq * seq


scal = {}
for T_text in (100, 300, 600):
    for V0 in (256, 576, 729):
        a = llm_fwd_flops(V0 + T_text, LLM["7B"])
        b = llm_fwd_flops(4 * V0 + T_text, LLM["7B"])
        scal[f"T={T_text},V0={V0}"] = round(b / a, 2)
out["llm_flops_ratio_when_patches_x4_7B"] = scal
# 어텐션만 보면 4배 토큰 -> 16배
out["attention_only_ratio_when_tokens_x4"] = 16

# 학습 시간 점검: Prism 7B (Controlled) < 9 시간 / A100 8장
hours = {}
for name, V in (("CLIP 336 (576)", 576), ("DINOSigLIP 384 (729)", 729)):
    for T_text in (150, 300, 450):
        for mfu in (0.3, 0.4, 0.5):
            tokens = 665_000 * (V + T_text)
            vit_f = 2 * (vit_l + (vit_so if "Siglip" in name or "SigLIP" in name else 0)) * V * 665_000
            f = 6 * LLM["7B"]["N"] * tokens + vit_f  # 체크포인팅 재계산 제외
            f_ckpt = 8 * LLM["7B"]["N"] * tokens + vit_f
            hours[f"{name},T={T_text},MFU={mfu}"] = {"no_ckpt_h": round(f / (8 * 312e12 * mfu) / 3600, 1),
                                                    "with_ckpt_h": round(f_ckpt / (8 * 312e12 * mfu) / 3600, 1)}
out["train_hours_8xA100_665k_1epoch"] = hours

# 단일 단계 절감 20-25% 점검: 정렬 단계 558K x (순전파 + 활성 역전파) 대 미세조정 665K x (순전파 + 완전 역전파)
sav = {}
for s in (0.20, 0.25):
    r = s / (1 - s) * (665 / 558)  # 정렬 샘플 하나의 비용 / 미세조정 샘플 하나의 비용
    sav[f"saving={s}"] = {"implied_cost_ratio_c1_over_c2": round(r, 3),
                          "implied_stage2_seq_len_if_V=576,T1=20": round(4 * (576 + 20) / (6 * r), 0)}
sav["equal_cost_saving"] = round(558 / (558 + 665), 3)
out["single_stage_saving_check"] = sav
out["steps_check"] = {"665_298_samples_over_128": round(665298 / 128, 1), "1.25_epochs": round(665298 / 128 * 1.25, 0),
                      "1.5_epochs": round(665298 / 128 * 1.5, 0), "code_max_steps": [6500, 7800]}

# OpenVLA 224px 입력: 토큰, 연산
ov = {"visual_tokens_224": 256, "visual_tokens_384": 729,
      "llm_gflops_per_forward_7B_prompt20": {str(V): round(llm_fwd_flops(V + 20, LLM["7B"]) / 1e9, 0) for V in (256, 729)},
      "ratio_384_over_224": round(llm_fwd_flops(729 + 20, LLM["7B"]) / llm_fwd_flops(256 + 20, LLM["7B"]), 2)}
out["openvla_224_vs_384"] = ov

# Prism-SigLIP 13B 이상 징후
gap = {}
for nm, s, c in (("7B controlled", "Prism-SigLIP 7B (Controlled)", "Prism-CLIP 7B (Controlled)"), ("7B", "Prism-SigLIP 7B", "Prism-CLIP 7B"),
                 ("13B controlled", "Prism-SigLIP 13B (Controlled)", "Prism-CLIP 13B (Controlled)"), ("13B", "Prism-SigLIP 13B", "Prism-CLIP 13B")):
    sec = PR7 if "7B" in nm else PR13
    a, b = R(sec, s), R(sec, c)
    gap[nm] = {k: round(a[k] - b[k], 2) for k in ("TextVQA", "RefCOCO", "OCIDRef", "VizWiz")}
out["siglip_minus_clip_prism"] = gap


# ---------- 7. 본문 문장 점검 ----------
GROUPS = {"VQA(4)": ["VQAv2", "GQA", "VizWiz", "TextVQA"], "위치 지정(4)": ["RefCOCO", "RefCOCO+", "RefCOCOg", "OCIDRef"],
          "도전 집합(4)": ["VSR", "POPE", "TallyQA", "AI2D"]}
fus = {}
for lab, b, a in (("SigLIP384 naive -> DINO+SigLIP naive", R(IP, "SigLIP ViT-SO 384px (Naive Resize)"), R(EN, "DINOv2 + SigLIP 384px (Naive Resize)")),
                  ("SigLIP384 letterbox -> DINO+SigLIP letterbox", R(IP, "SigLIP ViT-SO 384px (Letterbox)"), R(EN, "DINOv2 + SigLIP 384px (Letterbox)")),
                  ("CLIP336 naive -> DINO+CLIP naive", R(IP, "CLIP ViT-L 336px (Naive Resize)"), R(EN, "DINOv2 + CLIP 336px (Naive Resize)"))):
    d = vec(a) - vec(b)
    fus[lab] = {g: round(float(np.mean([d[B12.index(x)] for x in xs])), 2) for g, xs in GROUPS.items()}
    fus[lab]["TextVQA"] = round(float(d[B12.index("TextVQA")]), 2)
out["fusion_gain_by_group_pts"] = fus

# Prism이 같은 규모의 공식 LLaVA v1.5와 InstructBLIP을 모든 벤치마크에서 이기는가 ("strictly outperform", "uniformly outperform")
lose = {}
for sec, scale in ((PR7, "7B"), (PR13, "13B")):
    ref = vec(R(OPT, f"LLaVa v1.5 {scale}"))
    ib = vec(R(OPT, f"InstructBLIP {scale}"))
    for (s2, name), m in ROWS.items():
        if s2 != sec:
            continue
        v = vec(m)
        loses = [B12[i] for i in range(12) if v[i] < ref[i]]
        lose_ib = [B12[i] for i in range(12) if not np.isnan(ib[i]) and v[i] < ib[i]]
        lose[name] = {"loses_to_official_LLaVA": loses, "loses_to_InstructBLIP": lose_ib}
out["prism_vs_official_losses"] = lose

(HERE / "recompute_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# ---------- 요약 출력 ----------
print("항등 행:", out["identity_rows"])
print("VSR:", out["vsr"])
print("\n벤치마크 문항 수/MDE")
for b, v in bench.items():
    print(f"  {b:9s} n={v['n']:6d} base={v['base_acc']:.2f} wilson±{v['wilson95_halfwidth_pts']:.2f} MDE(q=.2)={v['mde_pts']['q=0.2']:.2f}")
print("\n축별")
for k, v in res.items():
    print(f"{k}: 쌍 {v['n_pairs']}, 평균 z차 {v['mean_z_diff']:+.3f}, +{v['pos']}/-{v['neg']} 부호검정 p={v['sign_p_two_sided_flat']}, "
          f"논문 p={v['paper_p']}, flat-t p={v['p_alt_gt_flat_t']}, Welch p={v['p_alt_gt_welch']}, wilcoxon={v['p_alt_gt_wilcoxon']}, "
          f"LOO안정={v['loo_mean_z_sign_stable']}, VSR뺀 z차={v['mean_z_diff_without_VSR']}")
print("\n노이즈 하한(공식 대 재현):", json.dumps(nf, ensure_ascii=False)[:1200])
print("\nMC FPR:", mc)
print("\n프로젝터 파라미터:", json.dumps(proj, ensure_ascii=False))
print("ViT 파라미터:", out["vit_params_analytic"])
print("해상도 4배 패치 시 LLM FLOPs 배율:", scal)
print("학습 시간:", {k: v for k, v in hours.items() if "T=300" in k})
print("단일단계:", sav, out["steps_check"])
print("OpenVLA:", ov)
print("SigLIP-CLIP 격차:", gap)
print("융합 이득:", fus)
print("Prism 패배:", lose)
