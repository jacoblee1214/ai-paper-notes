"""고노름 토큰 측정: DINOv2 ViT-S/14 (원본, register 4개), CLIP ViT-L/14 비전 타워.

Darcet et al. (arXiv:2309.16588)의 정의를 따라 마지막 층 출력(최종 정규화 앞)의 패치 토큰 노름을 본다.
(CLS와 register는 제외. 최종 정규화 뒤의 값은 보조로 함께 적는다.)

필요: torch, transformers, numpy, scipy, matplotlib, pillow. 가중치는 허깅페이스 캐시(오프라인)에서 읽는다.
이미지: MMVP_VLM (HF 데이터셋 `MMVP/MMVP_VLM`)의 images/ 폴더 270장. 경로는 인자로 준다.
실행: HF_HUB_OFFLINE=1 python3 high_norm_tokens.py [--images ../eyes_wide_shut/data/MMVP_VLM]   (이 폴더에서)
결과: high_norm_tokens_results.json, ../figures/own_high_norm_tokens.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
for name in ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

MODELS = {
    "dinov2_s": dict(label="DINOv2 ViT-S/14", hf="facebook/dinov2-small", reg=0,
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "dinov2_s_reg": dict(label="DINOv2 ViT-S/14 + register 4", hf="facebook/dinov2-with-registers-small", reg=4,
                         mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "clip_l": dict(label="CLIP ViT-L/14 (OpenAI)", hf="openai/clip-vit-large-patch14", reg=0,
                   mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758)),
}
RES = 224  # 두 계열 모두 14 픽셀 패치 -> 16x16 = 256 패치
KS = [2, 3, 4, 5, 7, 10]  # 층 중앙값의 몇 배를 넘으면 고노름으로 볼지(민감도)
K_MAIN = 3  # 로그 히스토그램의 골(중앙값의 약 2.5-3배)에 맞춤


def load_images(root: Path):
    d = root / "images" if (root / "images").is_dir() else root
    paths = sorted(d.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))
    if not paths:
        raise SystemExit(f"no jpg in {d}")
    return paths


def to_tensor(path, mean, std):
    im = Image.open(path).convert("RGB").resize((RES, RES), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    gray = a.mean(-1)
    t = (a - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.from_numpy(t).permute(2, 0, 1), gray


def patch_pixel_std(gray, g=16):
    p = RES // g
    return gray.reshape(g, p, g, p).transpose(0, 2, 1, 3).reshape(g * g, -1).std(-1)


def neighbor_cos(feat, g=16):
    """패치 토큰 (256, D)의 4-이웃 평균과의 코사인 유사도."""
    f = torch.nn.functional.normalize(feat, dim=-1).reshape(g, g, -1)
    pad = torch.nn.functional.pad(f.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")[0].permute(1, 2, 0)
    nb = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    nb = torch.nn.functional.normalize(nb, dim=-1)
    return (f * nb).sum(-1).reshape(-1)


def load_model(key):
    from transformers import CLIPVisionModel, Dinov2Model, Dinov2WithRegistersModel
    cls = {"dinov2_s": Dinov2Model, "dinov2_s_reg": Dinov2WithRegistersModel, "clip_l": CLIPVisionModel}[key]
    m = cls.from_pretrained(MODELS[key]["hf"], attn_implementation="eager").eval()
    return m


@torch.no_grad()
def run_model(key, paths):
    cfg = MODELS[key]
    m = load_model(key)
    nreg = cfg["reg"]
    final_ln = m.layernorm if key.startswith("dinov2") else m.post_layernorm
    norms, norms_ln, pix, ncos, cls_attn, reg_norm, cls_norm, reg_attn = [], [], [], [], [], [], [], []
    for i, p in enumerate(paths):
        x, gray = to_tensor(p, cfg["mean"], cfg["std"])
        out = m(pixel_values=x[None], output_hidden_states=True, output_attentions=True)
        hs = torch.stack(out.hidden_states)[:, 0]  # (L+1, T, D)
        pt = hs[:, 1 + nreg:]  # 패치 토큰
        norms.append(pt.norm(dim=-1).numpy())  # (L+1, 256)
        last = hs[-1]
        norms_ln.append(final_ln(last)[1 + nreg:].norm(dim=-1).numpy())
        pix.append(patch_pixel_std(gray))
        ncos.append(neighbor_cos(last[1 + nreg:]).numpy())
        cls_norm.append(last[0].norm().item())
        if nreg:
            reg_norm.append(last[1:1 + nreg].norm(dim=-1).numpy())
        a = out.attentions[-1][0].mean(0)[0]  # 마지막 층, 헤드 평균, CLS 행 (T,)
        cls_attn.append(a[1 + nreg:].numpy())
        if nreg:
            reg_attn.append(a[1:1 + nreg].sum().item())
        if (i + 1) % 90 == 0:
            print(key, i + 1, "/", len(paths), flush=True)
    return dict(
        norms=np.stack(norms), norms_ln=np.stack(norms_ln), pix=np.stack(pix), ncos=np.stack(ncos),
        cls_attn=np.stack(cls_attn), cls_norm=np.array(cls_norm),
        reg_norm=np.stack(reg_norm) if nreg else None, reg_attn=np.array(reg_attn) if nreg else None,
    )


def analyze(r):
    N, L1, P = r["norms"].shape
    last = r["norms"][:, -1]  # (N, 256)
    med_img = np.median(last, axis=1, keepdims=True)
    med_all = float(np.median(last))
    out = dict(n_images=int(N), n_tokens=int(N * P), n_layers=int(L1 - 1))
    out["last_layer"] = dict(
        median=med_all, mean=float(last.mean()), max=float(last.max()), p99=float(np.percentile(last, 99)),
        max_over_median=float(last.max() / med_all),
        per_image_max_over_median_median=float(np.median(last.max(1) / med_img[:, 0])),
        per_image_max_over_median_p90=float(np.percentile(last.max(1) / med_img[:, 0], 90)),
    )
    ln = r["norms_ln"]
    out["last_layer_after_final_norm"] = dict(
        median=float(np.median(ln)), max_over_median=float(ln.max() / np.median(ln)),
        frac_gt_main_median=float((ln > K_MAIN * np.median(ln)).mean()))
    # 임계값 민감도: 전체 패치 노름 중앙값의 k배 초과 비율, 그리고 이미지별 중앙값 기준
    out["sensitivity"] = {
        f"{k}x": dict(frac_vs_global_median=float((last > k * med_all).mean()),
                      frac_vs_image_median=float((last > k * med_img).mean()),
                      images_with_any=float(((last > k * med_all).any(1)).mean()))
        for k in KS}
    # 로그 노름 히스토그램의 바닥(이봉 간극) 탐색: 상위 꼬리의 가장 큰 간극
    ls = np.sort(np.log(last.reshape(-1)))
    gaps = np.diff(ls)
    top = slice(int(len(ls) * 0.9), None)
    gi = int(np.argmax(gaps[top])) + int(len(ls) * 0.9)
    out["largest_gap_in_top10pct"] = dict(
        lo=float(np.exp(ls[gi])), hi=float(np.exp(ls[gi + 1])), ratio=float(np.exp(gaps[gi])),
        typical_gap_ratio_top10pct=float(np.exp(np.median(gaps[top]))))
    # 층별 등장: 층 중앙값의 K_MAIN배 초과 비율과 최대/중앙값
    layer = []
    for l in range(L1):
        x = r["norms"][:, l]
        m = np.median(x)
        layer.append(dict(layer=l, median=float(m), max_over_median=float(x.max() / m),
                          frac_gt_main=float((x > K_MAIN * m).mean())))
    out["per_layer"] = layer
    first = [d["layer"] for d in layer if d["frac_gt_main"] >= 0.005]
    out["emergence_layer_frac_ge_0.5pct"] = first[0] if first else None
    # 고노름 집합 정의: 마지막 층에서 전역 중앙값의 K_MAIN배 초과
    hi = last > K_MAIN * med_all
    n_hi = int(hi.sum())
    out["high_norm_main"] = dict(k=K_MAIN, n=n_hi, frac=float(hi.mean()))
    # 대조용: 상위 2% 노름 토큰(Darcet의 약 2%와 같은 크기의 집합)
    thr2 = np.percentile(last, 98)
    top2 = last >= thr2
    out["top2pct_threshold_norm"] = float(thr2)

    def bg(mask, tag):
        d = {}
        if mask.sum() < 5:
            return dict(n=int(mask.sum()), note="표본 부족")
        # 이미지 안에서의 픽셀 표준편차 백분위(0.5 = 경향 없음)
        rk = np.stack([stats.rankdata(r["pix"][i]) / P for i in range(len(r["pix"]))])
        d["n"] = int(mask.sum())
        d["pixel_std_percentile_in_image_mean"] = float(rk[mask].mean())
        d["pixel_std_mean_high"] = float(r["pix"][mask].mean())
        d["pixel_std_mean_rest"] = float(r["pix"][~mask].mean())
        d["neighbor_cos_mean_high"] = float(r["ncos"][mask].mean())
        d["neighbor_cos_mean_rest"] = float(r["ncos"][~mask].mean())
        # 이미지 간 편향을 줄이는 검정: 이미지별 평균 차이의 부호 검정 대신 Mann-Whitney
        d["mannwhitney_p_pixel_std_less"] = float(
            stats.mannwhitneyu(r["pix"][mask], r["pix"][~mask], alternative="less").pvalue)
        # CLS->패치 attention 집중: 집합의 attention 질량 / 토큰 비율
        a = r["cls_attn"]
        d["cls_attn_mass_share"] = float(a[mask].sum() / a.sum())
        d["token_share"] = float(mask.mean())
        d["cls_attn_enrichment"] = d["cls_attn_mass_share"] / d["token_share"]
        return d

    out["background_main"] = bg(hi, "main")
    out["background_top2pct_control"] = bg(top2, "top2")
    out["cls_norm_median"] = float(np.median(r["cls_norm"]))
    if r["reg_norm"] is not None:
        rn = r["reg_norm"]
        out["registers"] = dict(
            norm_per_register_median=[float(np.median(rn[:, j])) for j in range(rn.shape[1])],
            norm_over_patch_median=[float(np.median(rn[:, j]) / med_all) for j in range(rn.shape[1])],
            cls_to_register_attention_mass_mean=float(r["reg_attn"].mean()),
            register_token_share=float(rn.shape[1] / (rn.shape[1] + P + 1)))
    return out, hi, top2


def plot(data, res):
    keys = list(MODELS)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {"dinov2_s": "#1b6ca8", "dinov2_s_reg": "#2e8b57", "clip_l": "#c0392b"}
    for k in keys:
        last = data[k]["norms"][:, -1].reshape(-1)
        med = np.median(last)
        ax[0].hist(np.log10(last / med), bins=100, histtype="step", color=cols[k], lw=1.6,
                   label=f"{MODELS[k]['label']} (중앙값 {med:.1f})")
        ax[1].plot([d["layer"] for d in res[k]["per_layer"]], [d["max_over_median"] for d in res[k]["per_layer"]],
                   color=cols[k], marker="o", ms=3, label=MODELS[k]["label"])
    ax[0].set_yscale("log")
    ax[0].axvline(np.log10(K_MAIN), color="k", ls=":", lw=1)
    ax[0].text(np.log10(K_MAIN) + 0.02, 1.5, f"기준 {K_MAIN}배", fontsize=8)
    ax[0].set_xlabel("log10 (패치 노름 / 모델 중앙값), 마지막 층 출력")
    ax[0].set_ylabel("토큰 수 (로그)")
    ax[0].set_title("마지막 층 패치 토큰 노름 분포")
    ax[0].legend(fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("층")
    ax[1].set_ylabel("최대 / 중앙값 (층별)")
    ax[1].set_title("층별 노름 최대/중앙값")
    ax[1].legend(fontsize=8)
    K_MAIN_IDX = KS.index(K_MAIN)
    w = 0.27
    for j, k in enumerate(keys):
        fr = [res[k]["sensitivity"][f"{kk}x"]["frac_vs_global_median"] * 100 for kk in KS]
        ax[2].bar(np.arange(len(KS)) + (j - 1) * w, fr, w, color=cols[k], label=MODELS[k]["label"])
    ax[2].set_xticks(range(len(KS)))
    ax[2].set_xticklabels([f"{k}배" for k in KS])
    ax[2].set_xlabel("임계값 (중앙값의 몇 배 초과)")
    ax[2].set_ylabel("초과 토큰 비율 (%)")
    ax[2].set_title("임계값 민감도")
    ax[2].legend(fontsize=8, loc="upper right")
    fig.suptitle("고노름 패치 토큰 측정 (이미지 270장, 224x224, 패치 256개/장)", y=1.0)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "own_high_norm_tokens.png", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="../eyes_wide_shut/data/MMVP_VLM", help="MMVP_VLM 폴더(images/ 포함)")
    args = ap.parse_args()
    root = (HERE.parent / args.images).resolve() if not Path(args.images).is_absolute() else Path(args.images)
    paths = load_images(root)
    print(len(paths), "images")
    torch.set_grad_enabled(False)
    data, res = {}, {}
    for k in MODELS:
        data[k] = run_model(k, paths)
        res[k], _, _ = analyze(data[k])
        res[k]["label"] = MODELS[k]["label"]
        res[k]["hf_id"] = MODELS[k]["hf"]
    res["_meta"] = dict(resolution=RES, grid="16x16", n_images=len(paths), k_main=K_MAIN,
                        definition="마지막 층 출력(최종 정규화 앞)의 패치 토큰 L2 노름, CLS·register 제외",
                        torch=torch.__version__)
    (HERE / "high_norm_tokens_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(data, res)
    for k in MODELS:
        s = res[k]
        print(k, "median %.1f max/med %.1f frac>3x %.4f%% emerg %s" % (
            s["last_layer"]["median"], s["last_layer"]["max_over_median"],
            100 * s["sensitivity"]["3x"]["frac_vs_global_median"], s["emergence_layer_frac_ge_0.5pct"]))


if __name__ == "__main__":
    main()
