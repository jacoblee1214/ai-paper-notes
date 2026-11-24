"""results_seed*.json -> ../figures/own_*.png  (모두 우리가 그린 그래프. 논문 그림은 쓰지 않는다)
실행: python plot_figures.py  (analysis/ 안에서)"""
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
for f in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/Library/Fonts/AppleSDGothicNeo.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

R = [json.load(open(p)) for p in sorted(glob.glob(str(HERE / "results_seed*.json")))]
SEEDS = [r["seed"] for r in R]
POS_LAB = ["채움", "채움", "주어 첫 토큰", "주어 마지막 토큰", "관계(마지막) 토큰"]
C_ = dict(a="#1f77b4", b="#d95f02", c="#1b9e77", d="#7570b3")


def mean_grid(key, src="trace_gn3"):
    return np.mean([np.array(r[src]["aie"][key]) for r in R], axis=0)  # (L,T)


def fig_heatmaps():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, (key, title) in zip(axs, [("resid", "잔차 상태 h"), ("mlp_w3", "MLP 출력 (층 3개 창)"), ("attn_w3", "attention 출력 (층 3개 창)")]):
        g = mean_grid(key).T  # (T, L)
        im = ax.imshow(g[2:], aspect="auto", cmap="Purples", vmin=0, vmax=max(0.05, g.max()))
        ax.set_yticks(range(3)); ax.set_yticklabels(POS_LAB[2:])
        ax.set_xticks(range(g.shape[1])); ax.set_xlabel("층")
        for i in range(3):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g[2 + i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if g[2 + i, j] > 0.5 * g.max() else "black")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"인과 추적 AIE (잡음 3σ, 사실 200개 x 잡음 10회, 시드 {len(R)}개 평균; 채움 토큰은 효과 0이라 생략)")
    fig.savefig(FIG / "own_trace_heatmaps.png", dpi=150)
    plt.close(fig)


def fig_lines():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True, sharey=True)
    L = len(R[0]["trace_gn3"]["aie"]["resid"])
    for ax, (pi, title) in zip(axs, [(3, "주어 마지막 토큰"), (4, "관계(마지막) 토큰")]):
        for key, col, lab in [("resid", C_["a"], "잔차 h"), ("mlp_w1", C_["b"], "MLP 1개 층"), ("mlp_w3", C_["c"], "MLP 3개 창"), ("attn_w3", C_["d"], "attention 3개 창")]:
            A = np.array([np.array(r["trace_gn3"]["aie"][key])[:, pi] for r in R])
            ax.plot(range(L), A.mean(0), color=col, label=lab, marker="o", ms=3)
            ax.fill_between(range(L), A.min(0), A.max(0), color=col, alpha=0.15)
        ax.set_title(title); ax.set_xlabel("층"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("AIE (정답 확률 회복)"); axs[0].legend(fontsize=8)
    fig.suptitle("층별 AIE: 평균과 시드별 최솟값-최댓값 범위")
    fig.savefig(FIG / "own_trace_lines.png", dpi=150)
    plt.close(fig)


def fig_sensitivity():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    mults = ["0.5", "1.0", "2.0", "3.0", "6.0"]
    ate = np.array([[r["trace_sweep"][m]["ate"] for m in mults] for r in R])
    axs[0].errorbar([float(m) for m in mults], ate.mean(0), yerr=ate.std(0), marker="o", color=C_["a"])
    axs[0].axhline(np.mean([r["trace_str"]["ate"] for r in R]), color=C_["b"], ls="--", label="다른 주어로 교체(STR)")
    axs[0].set_xlabel("잡음 크기 (임베딩 표준편차의 배수)"); axs[0].set_ylabel("ATE"); axs[0].set_title("총 효과 대 잡음 크기"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    # 잡음 배수별: 주어 마지막 토큰과 관계 토큰의 resid AIE의 층 평균 / ATE (정규화)
    for pi, col, lab in [(3, C_["c"], "주어 마지막 토큰"), (4, C_["d"], "관계 토큰")]:
        vals = []
        for m in mults:
            A = np.array([np.array(r["trace_sweep"][m]["aie"]["resid"])[:, pi] / max(r["trace_sweep"][m]["ate"], 1e-6) for r in R])
            vals.append(A.mean())
        axs[1].plot([float(m) for m in mults], vals, marker="o", color=col, label=lab)
    axs[1].set_xlabel("잡음 크기"); axs[1].set_ylabel("층 평균 AIE / ATE"); axs[1].set_title("복원 효과의 비율"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    L = len(R[0]["trace_gn3"]["aie"]["resid"])
    for src, col, lab in [("trace_gn3", C_["a"], "잡음 3σ"), ("trace_str", C_["b"], "주어 교체(STR)")]:
        A = np.array([np.array(r[src]["aie"]["mlp_w3"])[:, 3] / max(r[src]["ate"], 1e-6) for r in R])
        axs[2].plot(range(L), A.mean(0), marker="o", color=col, label=lab)
        axs[2].fill_between(range(L), A.min(0), A.max(0), color=col, alpha=0.15)
    axs[2].set_xlabel("층"); axs[2].set_title("주어 마지막 토큰, MLP 3개 창 AIE / ATE"); axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3)
    fig.savefig(FIG / "own_trace_sensitivity.png", dpi=150)
    plt.close(fig)


def fig_edit_layers():
    L = len(R[0]["trace_gn3"]["aie"]["resid"])
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True, sharey=True)
    for ax, (pos, title, pi) in zip(axs, [("last", "주어 마지막 토큰에서 편집", 3), ("final", "관계(마지막) 토큰에서 편집", 4), ("first", "주어 첫 토큰에서 편집", 2)]):
        for key, col, lab in [("es", C_["a"], "efficacy ES"), ("ps", C_["b"], "paraphrase PS"), ("nb_near", C_["c"], "이웃 정확도")]:
            A = np.array([[np.mean(r["sweep"][pos][str(l)][key]) for l in range(L)] for r in R])
            ax.errorbar(range(L), A.mean(0), yerr=A.std(0), marker="o", ms=3, color=col, label=lab, capsize=2)
        T = np.array([np.array(r["trace_gn3"]["aie"]["mlp_w1"])[:, pi] / max(r["trace_gn3"]["ate"], 1e-6) for r in R])
        ax.plot(range(L), T.mean(0), color="gray", ls="--", label="추적: MLP 1개 층 AIE/ATE")
        ax.set_title(title); ax.set_xlabel("편집 층"); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)
    axs[2].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f"ROME 편집 성공 대 편집 층·위치 (시드 {len(R)}개, 시드당 사실 {len(R[0]['edit_facts']['s'])}개의 평균과 시드 간 표준편차)")
    fig.savefig(FIG / "own_edit_layers.png", dpi=150)
    plt.close(fig)


def fig_ablation():
    names = list(R[0]["ablation"].keys())
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True, sharey=True)
    specs = [("pm", "새 목적어 확률 P[o*] (편집 문장)"), ("pp", "P[o*] (다른 표현 B)"), ("d_nb_prob", "이웃 정답 확률 감소"), ("d_glob_acc", "전역 정확도 감소")]
    for ax, (k, t) in zip(axs, specs):
        M = np.array([[np.mean(r["ablation"][n][k]) for n in names] for r in R])
        ax.barh(range(len(names)), M.mean(0), xerr=M.std(0), color=C_["a"])
        for i, v in enumerate(M.mean(0)):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
        ax.set_title(t, fontsize=10); ax.grid(alpha=0.3, axis="x")
    axs[0].set_yticks(range(len(names))); axs[0].set_yticklabels(names, fontsize=8); axs[0].invert_yaxis()
    site = R[0]["ablation_site"]
    fig.suptitle(f"구성 요소 제거 (시드 0의 가장 잘 듣는 자리: {site['pos']}, 층 {site['layer']}; 시드 평균)")
    fig.savefig(FIG / "own_ablation.png", dpi=150)
    plt.close(fig)


def fig_sequential():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for r in R:
        n = [t["n_edits"] for t in r["sequential"]]
        axs[0].plot(n, [t["edited_retained"] for t in r["sequential"]], marker="o", ms=3, label=f"시드 {r['seed']}")
        axs[1].plot(n, [t["unedited_acc"] for t in r["sequential"]], marker="o", ms=3, label=f"시드 {r['seed']}")
    axs[0].set_title("지금까지 편집한 사실 중 새 목적어를 내는 비율"); axs[1].set_title("편집하지 않은 사실의 정확도")
    for ax in axs:
        ax.set_xlabel("누적 편집 수"); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    axs[0].legend(fontsize=8)
    fig.savefig(FIG / "own_sequential.png", dpi=150)
    plt.close(fig)


def fig_localization_vs_edit():
    """사실별(fact x layer) 추적 효과 대 편집 성공(P[o*])의 산점도와 R^2 (Hase et al. 2023의 분석 방식)."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    L = len(R[0]["trace_gn3"]["aie"]["resid"])
    out = {}
    for pos, pi, col in [("final", 4, C_["b"]), ("last", 3, C_["a"])]:
        x, y, lay = [], [], []
        for r in R:
            te = np.array(r["tracing_per_fact"]["te"])
            ie = np.array(r["tracing_per_fact"]["mlp_w1"])[:, :, pi]  # (fact, layer)
            frac = ie / np.maximum(te[:, None], 1e-3)
            for l in range(L):
                x += frac[:, l].tolist()
                y += r["sweep"][pos][str(l)]["pm"]
                lay += [l] * len(frac[:, l])
        x, y, lay = map(np.array, (x, y, lay))
        axs[0 if pos == "final" else 1].scatter(x, y, s=8, alpha=0.4, color=col)
        rho = np.corrcoef(x, y)[0, 1]
        Xl = np.eye(L)[lay]
        r2 = lambda X: 1 - ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2_layer = r2(Xl)
        r2_trace = r2(np.c_[np.ones(len(x)), x])
        r2_both = r2(np.c_[Xl, x])
        out[pos] = dict(rho=float(rho), r2_layer=float(r2_layer), r2_trace=float(r2_trace), r2_both=float(r2_both), n=int(len(x)))
        axs[0 if pos == "final" else 1].set_title(f"{'관계 토큰' if pos == 'final' else '주어 마지막 토큰'}: 상관 {rho:.2f}")
        axs[0 if pos == "final" else 1].set_xlabel("사실별 추적 효과 (MLP 1개 층 IE / TE)"); axs[0 if pos == "final" else 1].set_ylabel("편집 후 P[o*]"); axs[0 if pos == "final" else 1].grid(alpha=0.3)
    w = 0.25
    for i, (pos, lab) in enumerate([("final", "관계 토큰"), ("last", "주어 마지막 토큰")]):
        axs[2].bar(np.arange(3) + i * w, [out[pos]["r2_layer"], out[pos]["r2_trace"], out[pos]["r2_both"]], w, label=lab)
    axs[2].set_xticks(np.arange(3) + w / 2); axs[2].set_xticklabels(["층만", "추적 효과만", "층 + 추적"]); axs[2].set_ylabel("R²"); axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3, axis="y")
    axs[2].set_title("편집 성공을 얼마나 설명하나")
    fig.savefig(FIG / "own_localization_vs_edit.png", dpi=150)
    plt.close(fig)
    (HERE / "localization_vs_edit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def fig_aux():
    """변형 B(주어 마지막 토큰의 보조 속성 손실): 기본 토이와 같은 AIE 줄 그래프와 편집 성공."""
    A_ = [json.load(open(p)) for p in sorted(glob.glob(str(HERE / "results_aux_seed*.json")))]
    if not A_:
        return
    L = len(A_[0]["trace_gn3"]["aie"]["resid"])
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6), constrained_layout=True)
    for ax, (pi, title) in zip(axs[:2], [(3, "주어 마지막 토큰"), (4, "관계(마지막) 토큰")]):
        for runs, ls, tag in [(R, "-", "기본"), (A_, "--", "보조 손실 변형")]:
            for key, col, lab in [("resid", C_["a"], "잔차 h"), ("mlp_w3", C_["c"], "MLP 3개 창"), ("attn_w3", C_["d"], "attention 3개 창")]:
                M = np.array([np.array(r["trace_gn3"]["aie"][key])[:, pi] for r in runs]).mean(0)
                ax.plot(range(L), M, ls, color=col, marker="o", ms=3, label=f"{lab} ({tag})")
        ax.set_title(title); ax.set_xlabel("층"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("AIE")
    axs[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for ax, (pos, title) in zip(axs[2:], [("last", "주어 마지막 토큰에서 편집"), ("final", "관계 토큰에서 편집")]):
        for runs, ls, tag in [(R, "-", "기본"), (A_, "--", "보조")]:
            for key, col, lab in [("es", C_["a"], "ES"), ("ps", C_["b"], "PS"), ("same_subj", C_["d"], "같은 주어 다른 관계 정확도"), ("nb_near", C_["c"], "이웃 정확도")]:
                M = np.array([[np.mean(r["sweep"][pos][str(l)][key]) for l in range(L)] for r in runs]).mean(0)
                ax.plot(range(L), M, ls, color=col, marker="o", ms=3, label=f"{lab} ({tag})")
        ax.set_title(title); ax.set_xlabel("편집 층"); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)
    axs[3].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(FIG / "own_aux_variant.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_heatmaps(); fig_lines(); fig_sensitivity(); fig_edit_layers(); fig_ablation(); fig_sequential(); fig_aux()
    print(fig_localization_vs_edit())
