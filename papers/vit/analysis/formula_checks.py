"""식·산술 검산: 파라미터 수(표 1), 토큰 수, attention 대 MLP 연산 비율, 표 6의 exaFLOPs, 위치 임베딩 보간 오차.
실행: python analysis/formula_checks.py  (논문 폴더 vit/ 에서, CPU, 수 초) -> formula_results.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as M  # noqa: E402

HERE = Path(__file__).resolve().parent
out = {}

# 1. 파라미터 수. 해석식(1000 클래스 헤드, 224 입력)과 표 1
table1 = {"B": 86, "L": 307, "H": 632}
rows = []
for name, patch in [("B", 16), ("L", 16), ("H", 14), ("B", 32), ("L", 32)]:
    c = M.CONFIGS[name]
    n = M.count_params_formula(c["depth"], c["dim"], c["mlp"], patch, 224, 1000)
    n_nohead = n - (c["dim"] * 1000 + 1000)
    rows.append(dict(model=f"{name}/{patch}", params_M=round(n / 1e6, 2), params_no_head_M=round(n_nohead / 1e6, 2),
                     table1_M=table1[name]))
out["params"] = rows

# 구현으로 만든 모델의 numel 과 대조(B/16 은 CPU 에서 만든다. L, H 는 메모리 때문에 해석식만)
m = M.make_vit("B/16", img_size=224, num_classes=1000)
numel = sum(p.numel() for p in m.parameters())
out["numel_B16"] = numel
out["formula_B16"] = M.count_params_formula(12, 768, 3072, 16, 224, 1000)
del m
for name, patch in [("L", 16), ("L", 32)]:
    c = M.CONFIGS[name]
    with torch.device("meta"):
        mm = M.make_vit(f"{name}/{patch}", img_size=224, num_classes=1000)
    out[f"numel_{name}{patch}_meta"] = sum(p.numel() for p in mm.parameters())
c = M.CONFIGS["H"]
with torch.device("meta"):
    mm = M.make_vit("H/14", img_size=224, num_classes=1000)
out["numel_H14_meta"] = sum(p.numel() for p in mm.parameters())

# 2. 해상도별 토큰 수와 attention/MLP 비율
res_rows = []
for name, patch in [("B", 16), ("L", 16), ("H", 14)]:
    c = M.CONFIGS[name]
    for r in (224, 384, 512, 518, 1024):
        if r % patch:
            continue
        t = (r // patch) ** 2 + 1
        macs = M.macs_formula(c["depth"], c["dim"], c["mlp"], patch, r, 1000)
        attn_layer = 2 * t * t * c["dim"]
        lin_layer = t * (4 * c["dim"] ** 2 + 2 * c["dim"] * c["mlp"])
        res_rows.append(dict(model=f"{name}/{patch}", res=r, tokens=t, GMACs=round(macs / 1e9, 1),
                             attn_over_linear=round(attn_layer / lin_layer, 3),
                             attn_share_of_layer=round(attn_layer / (attn_layer + lin_layer), 3)))
out["resolution"] = res_rows

# 3. 표 6 exaFLOPs 재계산(ViT 행). JFT 303M장 x 에폭 x 6 x MAC
jft = 303e6
t6 = [("B", 32, 7, 55), ("B", 16, 7, 224), ("L", 32, 7, 196), ("L", 16, 7, 783), ("L", 16, 14, 1567),
      ("H", 14, 14, 4262)]
rows = []
for s, p, ep, table in t6:
    c = M.CONFIGS[s]
    macs = M.macs_formula(c["depth"], c["dim"], c["mlp"], p, 224, 1000)
    ex = M.pretrain_exaflops(macs, jft * ep)
    rows.append(dict(model=f"{s}/{p}", epochs=ep, GMACs=round(macs / 1e9, 2), exaflops_ours=round(ex, 1),
                     exaflops_table6=table, rel_err_pct=round(100 * (ex - table) / table, 2)))
out["table6_exaflops"] = rows

# 4. 훈련 시퀀스 길이 비교: 패치 크기 절반이면 토큰 4배, 연산 4배 이상
c = M.CONFIGS["B"]
m32 = M.macs_formula(c["depth"], c["dim"], c["mlp"], 32, 224, 1000)
m16 = M.macs_formula(c["depth"], c["dim"], c["mlp"], 16, 224, 1000)
out["B_patch32_to_16_macs_ratio"] = round(m16 / m32, 3)
out["B_patch32_to_16_tokens_ratio"] = round(((224 // 16) ** 2 + 1) / ((224 // 32) ** 2 + 1), 3)

# 5. 보간: 14x14 -> 24x24 -> 14x14 왕복 오차(임의 임베딩, 부드러운 임베딩)
torch.manual_seed(0)
rand = torch.randn(1, 1 + 196, 64)
yy, xx = torch.meshgrid(torch.linspace(0, 1, 14), torch.linspace(0, 1, 14), indexing="ij")
smooth_grid = torch.stack([torch.sin(3 * yy + k) * torch.cos(2 * xx + k) for k in range(64)], dim=-1).reshape(1, 196, 64)
smooth = torch.cat([torch.zeros(1, 1, 64), smooth_grid], dim=1)
rt = {}
for name, pe in [("random", rand), ("smooth", smooth)]:
    up = M.resize_pos_embed(pe, (14, 14), (24, 24))
    back = M.resize_pos_embed(up, (24, 24), (14, 14))
    rt[name] = float((back[:, 1:] - pe[:, 1:]).pow(2).mean().sqrt() / pe[:, 1:].pow(2).mean().sqrt())
out["interp_roundtrip_rel_rmse"] = rt


# 6. 다른 분석(paligemma)이 쓴 SigLIP So400m/14 의 값과 대조: 폭 1152, 27층, MLP 4304, 224 -> 256 토큰, [CLS]와 헤드 없음
n = M.count_params_formula(27, 1152, 4304, 14, 224, 0, pool="gap") - 0
macs = M.macs_formula(27, 1152, 4304, 14, 224, 0, pool="gap")
out["so400m_14_224"] = dict(params_M=round(n / 1e6, 1), GMACs=round(macs / 1e9, 1), tokens=256,
                            paligemma_params_M=412.4)

# 7. §4.4 "ViT 는 같은 성능에 약 2-4배 적은 연산" 을 표 6 숫자로 다시 따져 본다.
# 표 6(논문 부록 C)에서 옮겨 적은 값: (이름, 에폭, ImageNet, ReaL, CIFAR-10, CIFAR-100, Pets, Flowers, exaFLOPs)
T6 = {
    "vit": [("B/32", 7, 80.73, 86.27, 98.61, 90.49, 93.40, 99.27, 55), ("B/16", 7, 84.15, 88.85, 99.00, 91.87, 95.80, 99.56, 224),
            ("L/32", 7, 84.37, 88.28, 99.19, 92.52, 95.83, 99.45, 196), ("L/16", 7, 86.30, 89.43, 99.38, 93.46, 96.81, 99.66, 783),
            ("L/16", 14, 87.12, 89.99, 99.38, 94.04, 97.11, 99.56, 1567), ("H/14", 14, 88.08, 90.36, 99.50, 94.71, 97.11, 99.71, 4262)],
    "resnet": [("R50x1", 7, 77.54, 84.56, 97.67, 86.07, 91.11, 94.26, 50), ("R50x2", 7, 82.12, 87.94, 98.29, 89.20, 93.43, 97.02, 199),
               ("R101x1", 7, 80.67, 87.07, 98.48, 89.17, 94.08, 95.95, 96), ("R152x1", 7, 81.88, 87.96, 98.82, 90.22, 94.17, 96.94, 141),
               ("R152x2", 7, 84.97, 89.69, 99.06, 92.05, 95.37, 98.62, 563), ("R152x2", 14, 85.56, 89.89, 99.24, 91.92, 95.75, 98.75, 1126),
               ("R200x3", 14, 87.22, 90.15, 99.34, 93.53, 96.32, 99.04, 3306)],
}


def frontier_ratios(metric):
    import numpy as np
    res = sorted(((metric(r), r[-1]) for r in T6["resnet"]), key=lambda t: t[1])
    pareto, best = [], -1
    for acc, ef in res:  # exaFLOPs 가 늘 때 정확도가 늘어나는 점만
        if acc > best:
            pareto.append((acc, ef))
            best = acc
    accs = np.array([a for a, _ in pareto])
    logef = np.log10([e for _, e in pareto])
    rows = []
    for v in T6["vit"]:
        a = metric(v)
        if accs[0] <= a <= accs[-1]:
            ef_res = 10 ** np.interp(a, accs, logef)
            rows.append(dict(model=f"{v[0]} {v[1]}ep", acc=round(a, 2), vit_ef=v[-1], resnet_ef_same_acc=round(float(ef_res)),
                             ratio=round(float(ef_res) / v[-1], 2)))
        else:
            rows.append(dict(model=f"{v[0]} {v[1]}ep", acc=round(a, 2), vit_ef=v[-1], resnet_ef_same_acc=None,
                             ratio=None, note="ResNet 범위 밖"))
    return rows


out["compute_ratio_imagenet"] = frontier_ratios(lambda r: r[2])
out["compute_ratio_avg5"] = frontier_ratios(lambda r: (r[2] + r[4] + r[5] + r[6] + r[7]) / 5)

(HERE / "formula_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
