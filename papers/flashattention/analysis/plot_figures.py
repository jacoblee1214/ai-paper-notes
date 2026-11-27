"""results.json, gpt2_flash_check_results.json, bench_cpu_results.json 로 그래프 네 장을 ../figures/ 에 그린다."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
fig_dir = here.parent / "figures"
fig_dir.mkdir(exist_ok=True)
R = json.load(open(here / "results.json"))
G = json.load(open(here / "gpt2_flash_check_results.json"))
B = json.load(open(here / "bench_cpu_results.json"))
plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
C1, C2, GR = "#1f6fb4", "#d9660a", "#555555"

# 1. HBM 접근 수 대 N^2 d^2 / M
rows = R["E1_rows"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.array([r["N"] ** 2 * r["d"] ** 2 / r["M"] for r in rows]); y = np.array([r["flash_fwd"] for r in rows]); yb = np.array([r["flash_bwd"] for r in rows])
ax[0].loglog(x, y, "o", color=C1, ms=4, label="순전파 (Alg. 1)")
ax[0].loglog(x, yb, "s", color=C2, ms=4, label="역전파 (Alg. 4)")
xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
t = R["E1"]
ax[0].loglog(xs, t["two_term_flash_fwd"]["a"] * xs, "-", color=C1, lw=1, alpha=.6)
ax[0].loglog(xs, t["two_term_flash_bwd"]["a"] * xs, "-", color=C2, lw=1, alpha=.6)
ax[0].set_xlabel("N²d²/M"); ax[0].set_ylabel("HBM 읽기+쓰기 (원소 수)")
ax[0].set_title(f"78개 (N,d,M) 설정, 기울기 12.3 · 20.2 (식: 12 · 20)")
ax[0].legend(frameon=False)
for d, c, mk in ((64, C1, "o"), (128, C2, "s")):
    Ms = [16384, 32768, 65536, 98304]
    ax[1].plot([m / d**2 for m in Ms], [R["E2"]["ratios_std_over_flash"][f"d{d}_M{m}_N1024"]["fwd"] for m in Ms], mk + "-", color=c, label=f"d={d}")
ax[1].axhline(1, color=GR, lw=1); ax[1].axvline(3, color=GR, ls="--", lw=1)
ax[1].text(3.1, ax[1].get_ylim()[1] * 0.9, "M = 3d²", color=GR)
ax[1].set_xlabel("M / d²"); ax[1].set_ylabel("표준 / Flash 순전파 HBM 접근 비 (N=1024)")
ax[1].set_title("M이 3d² 아래면 FlashAttention이 접근을 줄이지 못함"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(fig_dir / "own_hbm_scaling.png", dpi=150); plt.close(fig)

# 2. 블록 희소
e3 = R["E3"]
fig, ax = plt.subplots(figsize=(5.2, 4))
for pat, c, mk in (("random", C1, "o"), ("local", C2, "s")):
    rr = [r for r in e3["rows"] if r["pattern"] == pat]
    ax.plot([r["s"] for r in rr], [r["fwd_over_dense"] for r in rr], mk, color=c, label={"random": "무작위 블록", "local": "국소 창"}[pat])
s = np.linspace(0, 1, 20)
ax.plot(s, (e3["intercept"] + e3["slope"] * s) / e3["dense_fwd"], "-", color=GR, lw=1, label="2Nd + s·상수 (정확한 직선)")
ax.set_xlabel("0이 아닌 블록 비율 s"); ax.set_ylabel("순전파 HBM / 밀집 Flash"); ax.set_title("블록 희소: 접근 수가 s에 선형 (Prop. 4)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fig_dir / "own_block_sparse.png", dpi=150); plt.close(fig)

# 3. fp16 오차 (GPT-2 실제 Q, K, V)
rows = G["rows"]
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ml = np.array([r["max_logit"] for r in rows])
ax.loglog(ml, [r["f16_naive"] for r in rows], "o", ms=3, alpha=.5, color=C2, label="naive (S를 fp16으로 저장)")
ax.loglog(ml, [r["f16_flash"] for r in rows], "s", ms=3, alpha=.5, color=C1, label="타일 (S를 온칩 fp32로)")
ax.set_xlabel("헤드의 최대 |로짓|"); ax.set_ylabel("fp64 정답 대비 최대 절대 오차")
ax.set_title("GPT-2 small 실제 Q,K,V 720개 헤드 (fp16 입력)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(fig_dir / "own_gpt2_fp16_error.png", dpi=150); plt.close(fig)

# 4. CPU 메모리와 시간
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for impl, c, mk, lab in (("naive", C2, "o", "naive"), ("sdpa", C1, "s", "PyTorch SDPA(CPU 타일 커널)"), ("tiled", GR, "^", "이 폴더 타일 구현(파이썬)")):
    rr = [r for r in B if r["impl"] == impl and r["mode"] == "fwd"]
    ax[0].loglog([r["N"] for r in rr], [r["rss_mb_delta"] for r in rr], mk + "-", color=c, label=lab)
    ax[1].loglog([r["N"] for r in rr], [r["ms_median"] for r in rr], mk + "-", color=c, label=lab)
ax[0].set_ylabel("RSS 증가분 (MB)"); ax[1].set_ylabel("순전파 시간 (ms)")
for a in ax:
    a.set_xlabel("시퀀스 길이 N (2 헤드, d=64, fp32)")
ax[0].set_title("메모리: naive는 N²로, 타일은 거의 평탄"); ax[1].set_title("시간"); ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(fig_dir / "own_cpu_bench.png", dpi=150); plt.close(fig)
