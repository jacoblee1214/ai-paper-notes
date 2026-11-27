"""우리가 그린 그래프. streaming_ppl_results.json, sink_token_toy_results.json -> ../figures/own_*.png"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open(HERE / "streaming_ppl_results.json"))
COL = {"dense": "#4d4d4d", 0: "#d62728", 1: "#ff7f0e", 2: "#bcbd22", 4: "#1f77b4", 8: "#2ca02c", "rc": "#9467bd"}
rng = np.random.RandomState(0)


def ppl_ratio(key, C, mode, s_a, s_b, bucket="post_all", n_boot=2000):
    """exp(평균 nll 차이) = perplexity 비. 스트림 단위 부트스트랩."""
    a = np.array(R[key]["configs"][f"C{C}|s{s_a}|{mode}"]["buckets"][bucket]["per_stream_nll"])
    b = np.array(R[key]["configs"][f"C{C}|s{s_b}|{mode}"]["buckets"][bucket]["per_stream_nll"])
    d = a - b
    bs = d[rng.randint(0, len(d), (n_boot, len(d)))].mean(1)
    return float(np.exp(d.mean())), [float(np.exp(np.percentile(bs, q))) for q in (2.5, 97.5)]


def curves(ax, key, C, mode, title, sinks=(0, 1, 4), dense=True):
    r = R[key]
    x = (np.arange(len(r["configs"][f"C{C}|s0|{mode}"]["curve_bin128"])) + 0.5) * 128
    if dense and "dense" in r:
        ax.plot(x, np.exp(r["dense"]["curve_bin128"]), color=COL["dense"], lw=2, label="dense")
    for s in sinks:
        ax.plot(x, np.exp(r["configs"][f"C{C}|s{s}|{mode}"]["curve_bin128"]), color=COL[s], lw=1.6,
                label=f"{s}+{C - s}" + (" (윈도만)" if s == 0 else ""))
    ax.axvline(C, color="k", ls=":", lw=1, label=f"t = C = {C} (캐시가 참)")
    if r["T"] > r["pre_ctx"]:
        ax.axvline(r["pre_ctx"], color="gray", ls="--", lw=1, label=f"사전학습 문맥 {r['pre_ctx']}")
    ax.set_yscale("log")
    ax.set_xlabel("스트림 위치 t (128 토큰 구간)")
    ax.set_ylabel("perplexity (구간 평균 nll의 exp)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def fig_curves():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    curves(ax[0], "p160", 512, "orig", "Pythia-160M, 캐시 C=512 (RoPE, 사전학습 문맥 2048)")
    curves(ax[1], "p70", 512, "orig", "Pythia-70M, C=512")
    curves(ax[2], "gpt2", 256, "orig", "GPT-2 small, C=256 (절대 위치, 원래 위치 유지, T=1024)", sinks=(0, 1, 4))
    fig.suptitle("스트림 위치별 perplexity (n=24 스트림 평균). 윈도만 쓰면 C를 넘는 순간부터 오르고, 싱크 1~4개가 이를 막는다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_streaming_curves.png", dpi=140)


def fig_sinks():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    cases = [("p14", 512, "orig", "Pythia-14M C=512"), ("p70", 512, "orig", "Pythia-70M C=512"),
             ("p160", 512, "orig", "Pythia-160M C=512"), ("gpt2", 256, "orig", "GPT-2 C=256 (원래 위치)")]
    for key, C, mode, name in cases:
        ss = [1, 2, 4, 8]
        vals = [ppl_ratio(key, C, mode, s, 4) for s in ss]
        ax[0].errorbar(ss, [v[0] for v in vals], yerr=[[v[0] - v[1][0] for v in vals], [v[1][1] - v[0] for v in vals]],
                       marker="o", capsize=3, label=name)
    ax[0].axhline(1, color="k", lw=0.8)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks([1, 2, 4, 8])
    ax[0].set_xticklabels(["1", "2", "4", "8"])
    ax[0].set_xlabel("싱크 토큰 수 s (캐시 크기 C 고정)")
    ax[0].set_ylabel("perplexity ÷ (s=4의 perplexity)")
    ax[0].set_title("싱크 수 1~8 (캐시가 찬 뒤, 95% 부트스트랩 CI)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    names, vals = [], []
    for key, C, mode, name in cases + [("p160", 1024, "orig", "Pythia-160M C=1024"), ("gpt2", 512, "orig", "GPT-2 C=512")]:
        names.append(name)
        vals.append(ppl_ratio(key, C, mode, 0, 4))
    y = np.arange(len(names))
    ax[1].barh(y, [v[0] for v in vals], color="#d62728")
    for yi, v in zip(y, vals):
        ax[1].errorbar(v[0], yi, xerr=[[v[0] - v[1][0]], [v[1][1] - v[0]]], color="k", capsize=3)
        ax[1].text(v[1][1] * 1.12, yi, f"{v[0]:.2f}×", va="center", fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xlim(1, 3000)
    ax[1].set_xlabel("윈도만(0+C) perplexity ÷ 4개 싱크(4+(C-4))")
    ax[1].set_title("싱크를 빼면 얼마나 터지는가 (모델마다 다름)", fontsize=10)
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "own_sink_count.png", dpi=140)


def fig_window():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, Cs, mode, title in [(ax[0], "p160", [256, 512, 1024], "orig", "Pythia-160M"), (ax[1], "gpt2", [128, 256, 512], "orig", "GPT-2 small (원래 위치)")]:
        vals = [ppl_ratio(key, C, mode, 0, 4) for C in Cs]
        a.errorbar(Cs, [v[0] for v in vals], yerr=[[v[0] - v[1][0] for v in vals], [v[1][1] - v[0] for v in vals]], marker="o", capsize=3, color="#d62728")
        a.set_xscale("log", base=2)
        a.set_yscale("log")
        a.set_xticks(Cs)
        a.set_xticklabels([str(c) for c in Cs])
        a.set_xlabel("캐시 크기 C")
        a.set_ylabel("윈도만 ÷ 4개 싱크 (perplexity 비)")
        a.set_title(title, fontsize=10)
        a.grid(alpha=0.3)
    fig.suptitle("윈도가 클수록 싱크를 뺀 손해가 줄어든다 (캐시가 찬 뒤 구간, 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_window_size.png", dpi=140)


def fig_gpt2_positions():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, C in zip(ax, [256, 512]):
        w = 0.35
        for j, mode in enumerate(["orig", "slot"]):
            vals = [R["gpt2"]["configs"][f"C{C}|s{s}|{mode}"]["buckets"]["post_all"]["ppl"] for s in (0, 1, 4)]
            a.bar(np.arange(3) + (j - 0.5) * w, vals, w, label={"orig": "원래 위치(길이 1024 이하만 가능)", "slot": "넣을 때의 순번 min(t, C-1)"}[mode])
        rc = R["gpt2"]["configs"].get(f"C{C}|recompute")
        if rc:
            a.axhline(rc["stride"]["post_all"]["ppl"], color=COL["rc"], ls="--", label="재계산 기준선(스트라이드 위치)")
        a.axhline(R["gpt2"]["dense"]["all"]["all"]["ppl"], color=COL["dense"], ls=":", label="dense(전 구간)")
        a.set_yscale("log")
        a.set_xticks(range(3))
        a.set_xticklabels(["0+%d" % C, "1+%d" % (C - 1), "4+%d" % (C - 4)])
        a.set_title(f"GPT-2 small, C={C}", fontsize=10)
        a.set_ylabel("perplexity (캐시가 찬 뒤)")
        a.grid(alpha=0.3, axis="y")
        a.legend(fontsize=7)
    fig.suptitle("학습된 절대 위치 모델: 위치가 K/V에 굳어 캐시 안에서 다시 매길 수 없다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_gpt2_positions.png", dpi=140)


def fig_toy():
    p = HERE / "sink_token_toy_results.json"
    if not p.exists():
        return
    T = json.load(open(p))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    keys = ["0+64", "1+63", "2+62", "4+60", "8+56"]
    for j, (variant, col) in enumerate([("vanilla", "#1f77b4"), ("sink", "#d62728")]):
        for seed in T["config"]["seeds"]:
            r = T["runs"][f"{variant}|seed{seed}"]["stream"]
            ax.plot(range(5), [r[k]["ppl_post"] for k in keys], marker="o", color=col, alpha=0.5 + 0.5 * seed,
                    label=f"{variant} (시드 {seed})")
    ax.set_xticks(range(5))
    ax.set_xticklabels(keys)
    ax.set_ylim(5, 8)
    ax.set_xlabel("캐시 구성 (x개 초기 토큰 + 최근 토큰, C=64). sink 모델의 첫 토큰은 싱크 토큰")
    ax.set_ylabel("바이트 수준 perplexity (캐시가 찬 뒤)")
    ax.set_title("토이(0.8M, 1200스텝): 사전학습 싱크 토큰", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_sink_token_toy.png", dpi=140)


if __name__ == "__main__":
    fig_curves()
    fig_sinks()
    fig_window()
    fig_gpt2_positions()
    fig_toy()
    print("done")
