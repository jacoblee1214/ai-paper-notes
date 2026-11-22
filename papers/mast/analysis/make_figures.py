"""우리가 그린 그래프 두 장. analysis/*_results.json을 읽는다. 실행: python3 -I analysis/make_figures.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "report", "figures")
os.makedirs(OUT, exist_ok=True)
for p in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

pdf = json.load(open(os.path.join(HERE, "iaa_pdf_vs_hf_results.json")))
mad = json.load(open(os.path.join(HERE, "mad_results.json")))

# 그림 1: κ 논문 보고값 대 재계산
labels = ["라운드 2", "라운드 3", "새 MAS 2개\n(14모드, 보고 0.79)"]
paper = [0.92, 0.84, 0.79]
mine = [pdf["variants"]["PDF Round 2 pooled"]["cohen"], pdf["variants"]["PDF Round 3 pooled"]["cohen"],
        pdf["variants"]["PDF Gen 새 MAS 2개(Magentic, OpenManus) pooled"]["cohen"]]
ci = [pdf["variants"]["PDF Round 2 pooled"]["trace_boot95"], pdf["variants"]["PDF Round 3 pooled"]["trace_boot95"],
      pdf["variants"]["PDF Gen 새 MAS 2개(Magentic, OpenManus) pooled"]["cell_boot95"]]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(3)
ax.bar(x - 0.18, paper, 0.34, label="논문 보고값 (v1/v2 본문, v3 0.79)", color="#9bb7d9")
ax.bar(x + 0.18, mine, 0.34, label="PDF 원표 재계산 (3쌍 평균 Cohen κ)", color="#1f5fae")
for i, (lo, hi) in enumerate(ci):
    ax.plot([i + 0.18, i + 0.18], [lo, hi], color="k", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05); ax.set_ylabel("κ")
ax.set_title("주석자 간 일치도: 보고값과 재계산\n(막대 위 선은 95% 구간, 라운드는 trace 재표집, 새 MAS는 셀 재표집)", fontsize=9)
ax.legend(fontsize=7.5, loc="lower left")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "own_kappa.png"), dpi=170); plt.close()

# 그림 2: 실패 모드 점유율 논문 그림 1 대 공개 LLM 라벨
modes = list(mad["fig1_stated"].keys())
f1 = [mad["fig1_stated"][m] for m in modes]
f2 = [mad["fig2_stated"][m] for m in modes]
dt = [100 * mad["mode_share_of_all_flags"][m] for m in modes]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(modes)); w = 0.28
ax.bar(x - w, f1, w, label="논문 그림 1 표기", color="#9bb7d9")
ax.bar(x, f2, w, label="논문 그림 2 표기", color="#c9c6bb")
ax.bar(x + w, dt, w, label="공개 데이터 재계산 (전체 flag 중 점유율)", color="#1f5fae")
ax.set_xticks(x); ax.set_xticklabels(modes, fontsize=8); ax.set_ylabel("%")
ax.set_title("실패 모드 점유율: 논문 표기와 공개 LLM 라벨 (1,638 trace)", fontsize=9)
ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "own_mode_share.png"), dpi=170); plt.close()
print("ok")
