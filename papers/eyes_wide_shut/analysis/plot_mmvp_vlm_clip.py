"""mmvp_vlm_clip_results.json -> ../report/figures/own_clip_margin.png (우리가 그린 그래프, matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
for f in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open(HERE / "mmvp_vlm_clip_results.json", encoding="utf-8"))
cfgs = {k: v for k, v in R["configs"].items() if "error" not in v}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# (a) 쌍별 결정 여유 |p-0.5| (fp32 CPU) 분포
m = np.array([d["margin"] for d in R["pair_detail_fp32_cpu"]])
ax[0].hist(m, bins=np.linspace(0, 0.5, 26), color="#4C78A8")
ax[0].set_yscale("log")
for t in (0.01, 0.05):
    ax[0].axvline(t, color="#E45756", ls="--", lw=1)
ax[0].set_xlabel("쌍의 결정 여유 min |p-0.5| (두 문장 중 작은 쪽)")
ax[0].set_ylabel("쌍 수 (로그)")
ax[0].set_title(f"fp32 CPU: 135쌍의 여유 분포 (|p-0.5|<0.01: {int((m<0.01).sum())}쌍)")
# (b) 패턴별 점수: 우리 fp32 CPU, fp16 MPS, 표 1
pats = [x["type"] for x in cfgs["fp32_cpu"]["per_pattern"]]
x = np.arange(len(pats)); w = 0.27
ax[1].bar(x - w, R["table1"], w, label="표 1", color="#E45756")
ax[1].bar(x, [q["acc"] for q in cfgs["fp32_cpu"]["per_pattern"]], w, label=f"우리 fp32 CPU/MPS (평균 {cfgs['fp32_cpu']['avg']:.1f})", color="#4C78A8")
ax[1].bar(x + w, [q["acc"] for q in cfgs["fp16_mps"]["per_pattern"]], w, label=f"우리 fp16 MPS (평균 {cfgs['fp16_mps']['avg']:.1f})", color="#72B7B2")
KO = {"Orientation": "방향", "Presence": "존재", "State": "상태", "Quantity": "수", "Spatial": "위치",
      "Color": "색", "Structural Character": "구조", "Text": "글자", "Camera Perspective": "시점"}
ax[1].set_xticks(x); ax[1].set_xticklabels([KO.get(q, q) for q in pats], fontsize=9)
ax[1].set_ylabel("패턴 점수 (%, 15쌍, 1쌍 = 6.7%p)")
ax[1].set_title("패턴별 점수: 표 1(평균 19.3) 대 우리 측정")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "own_clip_margin.png", dpi=200); plt.close(fig)
print("ok")
