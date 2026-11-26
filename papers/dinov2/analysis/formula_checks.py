"""논문 표와 식의 재계산, 식 유도의 수치 검산. 수 초, CPU. 결과는 formula_results.json.
실행: python3 formula_checks.py   (이 폴더에서)
옮겨 적은 표 값의 출처는 각 항목 주석(논문 v2 표 번호)이다.
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as M  # noqa: E402

out = {}

# ---------------------------------------------------------------- 1. 표 1 (iBOT -> DINOv2 절제) 증가분 재계산
# (이름, k-NN, 선형). 논문 표기 증가분은 같은 줄의 "↑/↓ x.x". 둘 다 인쇄값으로 비교.
t1 = [("iBOT", 72.9, 82.3, None, None),
      ("+our reproduction", 74.5, 83.2, 1.6, 0.9),
      ("+LayerScale, Stochastic Depth", 75.4, 82.0, 0.9, -1.2),
      ("+128k prototypes", 76.6, 81.9, 1.2, -0.1),
      ("+KoLeo", 78.9, 82.5, 2.3, 0.6),
      ("+SwiGLU FFN", 78.7, 83.1, -0.2, 0.6),
      ("+Patch size 14", 78.9, 83.5, 0.2, 0.4),
      ("+Teacher momentum 0.994", 79.4, 83.6, 0.5, 0.1),
      ("+Tweak warmup schedules", 80.5, 83.8, 1.1, 0.2),
      ("+Batch size 3k", 81.7, 84.7, 1.2, 0.9),
      ("+Sinkhorn-Knopp", 81.7, 84.7, 0.0, 0.0),
      ("+Untying heads = DINOv2", 82.0, 84.5, 0.3, -0.2)]
rows, mism = [], []
for i in range(1, len(t1)):
    dk, dl = round(t1[i][1] - t1[i - 1][1], 1), round(t1[i][2] - t1[i - 1][2], 1)
    rows.append(dict(step=t1[i][0], knn_delta=dk, lin_delta=dl, printed_knn=t1[i][3], printed_lin=t1[i][4]))
    if abs(dk - t1[i][3]) > 1e-9 or abs(dl - t1[i][4]) > 1e-9:
        mism.append(t1[i][0])
out["table1"] = dict(rows=rows, mismatches=mism, total_knn=round(t1[-1][1] - t1[0][1], 1),
                     total_lin=round(t1[-1][2] - t1[0][2], 1),
                     n_steps_with_lin_drop=sum(1 for r in rows if r["lin_delta"] < 0),
                     knn_resolution=0.1, note="한 번 실행, 시드·구간 없음. k-NN 기준 항목이 12개 중 1개(SwiGLU)만 하락")

# ---------------------------------------------------------------- 2. 파라미터 수 (백본만, 머리 제외)


def vit_params(dim, depth, ffn, patch=14, img=518, layerscale=True, in_ch=3, mlp_ratio=4):
    n_tok = (img // patch) ** 2
    if ffn == "mlp":
        h = mlp_ratio * dim
        ffn_p = dim * h + h + h * dim + dim
    else:  # swiglufused: hidden = (int(4D * 2/3) + 7) // 8 * 8, w12: D -> 2h, w3: h -> D
        h = (int(mlp_ratio * dim * 2 / 3) + 7) // 8 * 8
        ffn_p = dim * 2 * h + 2 * h + h * dim + dim
    per_layer = 2 * (2 * dim) + (3 * dim * dim + 3 * dim) + (dim * dim + dim) + ffn_p + (2 * dim if layerscale else 0)
    embed = patch * patch * in_ch * dim + dim
    return embed + dim + dim + n_tok * dim + depth * per_layer + 2 * dim  # patch, cls, mask_token, pos, 블록, 마지막 LN


cfg = {"ViT-S/14": (384, 12, "mlp"), "ViT-B/14": (768, 12, "mlp"), "ViT-B/14 (표 17의 18블록)": (768, 18, "mlp"),
       "ViT-L/14": (1024, 24, "mlp"), "ViT-g/14 (SwiGLU)": (1536, 40, "swiglufused")}
readme = {"ViT-S/14": 21e6, "ViT-B/14": 86e6, "ViT-L/14": 300e6, "ViT-g/14 (SwiGLU)": 1100e6}
out["params"] = {k: dict(params_M=round(vit_params(*v) / 1e6, 1), readme_M=(readme[k] / 1e6 if k in readme else None))
                 for k, v in cfg.items()}

# ---------------------------------------------------------------- 3. 탄소 (표 14)
mwh = 22016 * 0.4 * 1.1 / 1000
out["carbon"] = dict(mwh=round(mwh, 2), tco2=round(mwh * 0.385, 2), paper_mwh=9.7, paper_tco2=3.7,
                     openclip_L_ratio=round(22.4 / mwh, 2), openclip_G_ratio=round(118.9 / mwh, 2),
                     note="본문 '10x more'는 OpenCLIP-G(12.3배)에만 맞고 -L(2.3배)에는 아님")

# ---------------------------------------------------------------- 4. DINOv2 ViT-g/14 와 L/14(증류) 대 OpenCLIP ViT-G/14, 표 4, 6-11 옮겨 적음
# 각 항목: (이름, OpenCLIP-G, DINOv2 g, DINOv2 L, 낮을수록 좋음)
bench = {
    "image": [  # 표 4(선형 val/ReaL/V2, k-NN), 표 6(Im-A, Im-R, Im-C, Sketch), 표 7(6개), 표 8(12개)
        ("INet linear val", 86.2, 86.5, 86.3, 0), ("INet ReaL", 89.4, 89.6, 89.5, 0), ("INet V2", 77.2, 78.4, 78.0, 0), ("INet kNN", 83.2, 83.5, 83.5, 0),
        ("Im-A", 63.8, 75.9, 71.3, 0), ("Im-R", 87.8, 78.8, 74.4, 0), ("Im-C (lower)", 45.3, 28.2, 31.5, 1), ("Sketch", 66.4, 62.5, 59.3, 0),
        ("iNat18", 73.0, 81.6, 80.4, 0), ("iNat21", 76.0, 85.7, 85.1, 0), ("Places205", 69.8, 67.5, 67.3, 0),
        ("K400", 78.3, 78.4, 76.3, 0), ("UCF101", 90.7, 91.2, 90.5, 0), ("SSv2", 35.8, 38.3, 35.6, 0),
        ("Food", 94.5, 94.7, 94.3, 0), ("C10", 98.7, 99.5, 99.3, 0), ("C100", 91.0, 94.4, 93.4, 0), ("SUN", 84.0, 78.7, 78.7, 0),
        ("Cars", 96.1, 91.4, 90.1, 0), ("Aircraft", 80.2, 87.2, 81.5, 0), ("VOC", 89.3, 89.0, 88.3, 0), ("DTD", 86.0, 84.5, 84.0, 0),
        ("Pets", 95.7, 96.7, 96.6, 0), ("Caltech101", 98.1, 97.6, 97.5, 0), ("Flowers", 99.5, 99.7, 99.7, 0), ("CUB", 89.9, 91.6, 90.5, 0)],
    "instance_pixel": [  # 표 9(8열), 표 10(6열), 표 11(9열, RMSE 낮을수록 좋음)
        ("Oxford-M", 50.7, 73.6, 75.1, 0), ("Oxford-H", 19.7, 52.3, 54.0, 0), ("Paris-M", 79.2, 92.1, 92.7, 0), ("Paris-H", 60.2, 82.6, 83.5, 0),
        ("Met GAP", 6.5, 36.8, 40.0, 0), ("Met GAP-", 23.9, 73.6, 68.9, 0), ("Met ACC", 34.4, 76.5, 71.6, 0), ("AmsterTime", 24.6, 46.7, 50.0, 0),
        ("ADE lin", 39.3, 49.0, 47.7, 0), ("ADE +ms", 46.0, 53.0, 53.1, 0), ("City lin", 60.3, 71.3, 70.3, 0), ("City +ms", 70.3, 81.0, 80.9, 0),
        ("VOC lin", 71.4, 83.0, 82.1, 0), ("VOC +ms", 79.2, 86.2, 86.0, 0),
        ("NYU lin1", 0.541, 0.344, 0.384, 1), ("NYU lin4", 0.510, 0.298, 0.333, 1), ("NYU DPT", 0.414, 0.279, 0.293, 1),
        ("KITTI lin1", 3.57, 2.62, 2.78, 1), ("KITTI lin4", 3.21, 2.35, 2.50, 1), ("KITTI DPT", 2.56, 2.11, 2.14, 1),
        ("SUN lin1", 0.537, 0.402, 0.429, 1), ("SUN lin4", 0.476, 0.362, 0.396, 1), ("SUN DPT", 0.408, 0.338, 0.360, 1)],
}
res = {}
for grp, items in bench.items():
    for name, idx in (("g", 2), ("L_distilled", 3)):
        wins = [it[0] for it in items if (it[idx] < it[1] if it[4] else it[idx] > it[1])]
        loses = [it[0] for it in items if (it[idx] > it[1] if it[4] else it[idx] < it[1])]
        res[f"{grp}_{name}"] = dict(n=len(items), wins=len(wins), losses=len(loses), lose_items=loses)
out["vs_openclip_G"] = res

# ---------------------------------------------------------------- 5. 표 2(데이터), 표 3(소거), 그림 5(증류) 증가분
t2 = {"INet-22k": [85.9, 73.5, 46.6, 62.5, 81.1, 85.6, 67.0], "INet-22k\\1k": [85.3, 70.3, 46.2, 58.7, 80.1, 85.1, 66.5],
      "Uncurated": [83.3, 59.4, 48.5, 54.3, 68.0, 76.4, 67.2], "LVD-142M": [85.8, 73.9, 47.7, 64.6, 82.3, 86.4, 67.6]}
cols = ["INet-1k", "Im-A", "ADE-20k", "Oxford-M", "iNat18", "iNat21", "Places205"]
out["table2"] = {k: dict(wins=[c for c, a, b in zip(cols, t2["LVD-142M"], t2[k]) if a > b],
                         losses=[c for c, a, b in zip(cols, t2["LVD-142M"], t2[k]) if a < b])
                 for k in ("INet-22k", "Uncurated")}
out["table3"] = dict(koleo_oxfordM=round(63.9 - 55.6, 1), koleo_others=dict(INet1k=round(85.8 - 85.3, 1), ImA=round(72.8 - 70.6, 1), ADE=round(47.1 - 47.2, 1)),
                     mim_ade=round(47.1 - 44.2, 1), mim_oxfordM=round(63.9 - 64.3, 1), mim_imA=round(72.8 - 72.0, 1), mim_inet=round(85.8 - 85.3, 1))
fig5 = {"g scratch": [86.5, 73.4, 1.00, 92.1, 78.3, 75.2, 77.0, 69.3], "L scratch": [84.5, 72.2, 1.10, 90.2, 75.8, 71.3, 69.5, 67.3],
        "L distill": [86.3, 73.3, 1.08, 91.2, 77.6, 76.3, 74.5, 67.5]}
names5 = ["INet-1k", "Segm", "Depth RMSE (lower)", "Classif", "Finegr", "Retriev", "ARSketch", "Video"]
out["fig5"] = dict(names=names5, distill_minus_scratch=[round(a - b, 2) for a, b in zip(fig5["L distill"], fig5["L scratch"])],
                   distill_minus_g=[round(a - b, 2) for a, b in zip(fig5["L distill"], fig5["g scratch"])])

# ---------------------------------------------------------------- 6. ImageNet 표본 오차(50,000장), 0.3%p 차이
n = 50000
acc = 0.865
se = math.sqrt(acc * (1 - acc) / n)
out["imagenet_sampling"] = dict(n=n, acc=acc, se_single_pp=round(100 * se, 3), halfwidth95_single_pp=round(196 * se, 3),
                                paired_halfwidth95_pp={f"disagree {int(d * 100)}%": round(196 * math.sqrt(d / n), 3) for d in (0.05, 0.10, 0.15)},
                                observed_diff_pp=0.3, note="쌍 비교 구간은 두 모델이 서로 다르게 맞히는 비율(불일치율)을 가정한 값. 논문은 불일치율을 주지 않음")

# ---------------------------------------------------------------- 7. 해상도 224 -> 416(ViT-L/16) 연산비 (그림 6 "약 3배")


def macs(tokens, dim=1024, mlp=4096, depth=24):
    return depth * (tokens * (4 * dim * dim + 2 * dim * mlp) + 2 * tokens * tokens * dim)


t224, t416 = (224 // 16) ** 2 + 1, (416 // 16) ** 2 + 1
out["resolution_cost"] = dict(tokens_224=t224, tokens_416=t416, forward_ratio=round(macs(t416) / macs(t224), 2),
                              note="전역 크롭 한 장의 순전파 비. 지역 크롭(변화 없음)이 섞이면 학습 전체 비는 이보다 작아 '약 3배'와 모순되지 않음")


# ---------------------------------------------------------------- 7b. 멀티 크롭 비용(ViT-g/14: D=1536, 40층, SwiGLU 은닉 4096): 전역 2(224) + 지역 8(98)
def macs_g(tokens, dim=1536, depth=40, h=4096):
    # qkv 3D^2, 사영 D^2, SwiGLU 3 D h, 어텐션 2 T^2 D
    return depth * (tokens * (4 * dim * dim + 3 * dim * h) + 2 * tokens * tokens * dim)


tg, tl = (224 // 14) ** 2 + 1, (98 // 14) ** 2 + 1
c_multi = 2 * macs_g(tg) + 8 * macs_g(tl)
out["multicrop_cost"] = dict(tokens_global=tg, tokens_local=tl, local_over_global=round(macs_g(tl) / macs_g(tg), 3),
                             two_global_plus_8_local_over_2_global=round(c_multi / (2 * macs_g(tg)), 3),
                             vs_10_global=round(c_multi / (10 * macs_g(tg)), 3),
                             note="학생 순전파 기준(교사는 전역 2개만 순전파). 지역 크롭 8개를 더해도 전역 2개 대비 약 1.76배(10개 전부 전역이면 1/0.35 = 2.8배 비쌈)")

# ---------------------------------------------------------------- 8. 식 검산
# 8a. SK 반복 수와 열 합 오차(로짓 퍼짐별), B=256, K=32, T=0.05
torch.manual_seed(0)
sk = {}
for sd in (0.05, 0.1, 0.3, 1.0):
    z = torch.randn(256, 32) * sd
    sk[f"logit_std_{sd}"] = {str(n_it): round(float(((M.sinkhorn_knopp(z, 0.05, n_it).sum(0) / 256) * 32 - 1).abs().max()), 4)
                             for n_it in (1, 3, 10, 50, 200)}
out["sinkhorn_col_err"] = sk

# 8b. KL 엔트로피 추정량: N(0,I_d), n 별
kl = {}
for d in (2, 8):
    for n_pts in (500, 2000, 8000):
        torch.manual_seed(1)
        errs = [M.kl_entropy(torch.randn(n_pts, d)) - d / 2 * math.log(2 * math.pi * math.e) for _ in range(5)]
        kl[f"d{d}_n{n_pts}"] = dict(bias=round(sum(errs) / 5, 4), sd=round(float(torch.tensor(errs).std()), 4))
out["kl_entropy_error_vs_gaussian"] = kl

# 8c. 교사 엔트로피 = f(로짓 퍼짐 / 온도): 중심화한 로짓 z ~ N(0, s^2), K=64
torch.manual_seed(2)
ent = {}
for s in (0.005, 0.01, 0.02, 0.05, 0.1, 0.3):
    ent[f"spread_{s}"] = {}
    for t in (0.02, 0.04, 0.07, 0.1):
        p = torch.softmax(torch.randn(2000, 64) * s / t, dim=-1)
        ent[f"spread_{s}"][f"T_{t}"] = round(float(-(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()), 3)
out["entropy_vs_spread_over_T"] = dict(log_K=round(math.log(64), 3), table=ent)

# 8d. 손실 = H(p_t) + KL(p_t || p_s) (교차 엔트로피 분해)
torch.manual_seed(3)
pt = torch.softmax(torch.randn(100, 16) * 2, -1)
ps = torch.softmax(torch.randn(100, 16), -1)
ce = -(pt * ps.log()).sum(-1)
dec = -(pt * pt.log()).sum(-1) + (pt * (pt.log() - ps.log())).sum(-1)
out["ce_decomposition_max_err"] = float((ce - dec).abs().max())

# 8e. EMA 닫힌 꼴과 유효 창 길이 1/(1-m)
m = 0.994
out["ema"] = dict(m=m, effective_window=round(1 / (1 - m), 1), weight_on_last_k=dict(
    {f"k={k}": round(1 - m ** k, 3) for k in (10, 100, 167, 500, 1000)}),
    note="논문 m 은 0.994 에서 1 로 코사인(표 16, 625k 스텝). 초기 유효 창 약 167 스텝, 끝에서는 무한대")

Path(__file__).with_name("formula_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: out[k] for k in ("params", "carbon", "vs_openclip_G", "table2", "fig5", "imagenet_sampling", "resolution_cost", "multicrop_cost")}, indent=1, ensure_ascii=False))
print("table1 mismatches:", out["table1"]["mismatches"], out["table1"]["total_knn"], out["table1"]["total_lin"])
print("sk", json.dumps(out["sinkhorn_col_err"]))
print("kl", json.dumps(out["kl_entropy_error_vs_gaussian"]))
print("ent", json.dumps(out["entropy_vs_spread_over_T"]["table"]))
print("ce dec", out["ce_decomposition_max_err"])
