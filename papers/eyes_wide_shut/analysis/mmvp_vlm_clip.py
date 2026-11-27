"""MMVP-VLM(135쌍)을 OpenAI CLIP ViT-L/14(HF 변환본)로 평가하고 정밀도·장치별 판정 민감도를 잰다.

채점 규칙은 공식 evaluate_vlm.py와 같다: 쌍(홀수 qid=이미지1이 정답, 짝수 qid=이미지2)마다
두 문장 각각에서 softmax(텍스트->이미지 2개)의 이미지1 확률이 0.5 초과면 이미지1, 아니면 이미지2를 고르고,
두 문장 모두 맞아야 그 쌍을 맞힌 것으로 센다. 패턴 15쌍 정확도, 9패턴 평균.
사용: python mmvp_vlm_clip.py [--data ../data/MMVP_VLM] [--out mmvp_vlm_clip_results.json]
가중치는 로컬 HF 캐시의 openai/clip-vit-large-patch14만 쓴다(HF_HUB_OFFLINE=1).
"""
import argparse, csv, json, math, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL = "openai/clip-vit-large-patch14"
CATS = ["Orientation", "Presence", "State", "Quantity", "Spatial", "Color",
        "Structural Character", "Text", "Camera Perspective"]
TABLE1 = [13.3, 13.3, 20.0, 20.0, 13.3, 53.3, 20.0, 6.7, 13.3]  # 논문 표 1, OpenAI ViT-L-14 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
THRS = [0.001, 0.01, 0.02, 0.05, 0.075]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def openai_style_pixels(img):
    """OpenAI clip.load의 preprocess와 같은 순서: Resize(224, BICUBIC, 짧은 변) -> CenterCrop -> RGB -> [0,1] -> 정규화."""
    img = img.convert("RGB")
    w, h = img.size
    s = 224 / min(w, h)
    nw, nh = max(224, round(w * s)), max(224, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - 224) // 2, (nh - 224) // 2
    a = np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def load_pairs(data):
    rows = list(csv.reader(open(os.path.join(data, "Questions.csv"), encoding="utf-8")))[1:]
    pairs = []
    for i in range(0, len(rows), 2):
        (q1, t1, s1), (q2, t2, s2) = rows[i], rows[i + 1]
        assert t1 == t2 == CATS[(i // 2) // 15]
        pairs.append(dict(qid=(int(q1), int(q2)), type=t1, text=("a photo of " + s1, "a photo of " + s2),
                          img=tuple(os.path.join(data, "MLLM_VLM Images", t1, f"{q}.jpg") for q in (q1, q2))))
    return pairs


@torch.no_grad()
def run(model, proc, pairs, device, dtype, preproc):
    out = []
    for p in pairs:
        ims = [Image.open(f) for f in p["img"]]
        if preproc == "hf":
            px = proc(images=ims, return_tensors="pt")["pixel_values"]
        else:
            px = torch.stack([openai_style_pixels(i) for i in ims])
        tok = proc.tokenizer(list(p["text"]), padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        ie = model.get_image_features(pixel_values=px.to(device, dtype))
        te = model.get_text_features(input_ids=tok["input_ids"].to(device), attention_mask=tok["attention_mask"].to(device))
        ie, te = ie if torch.is_tensor(ie) else ie.pooler_output, te if torch.is_tensor(te) else te.pooler_output
        ie, te = ie / ie.norm(dim=-1, keepdim=True), te / te.norm(dim=-1, keepdim=True)
        logits_per_text = model.logit_scale.exp() * te @ ie.t()          # [2 문장, 2 이미지]
        probs = logits_per_text.float().softmax(dim=-1).cpu().numpy()    # 이미지 축 softmax(공식 코드와 같음)
        s1 = float(probs[0, 0]); s2 = float(probs[1, 0])                 # 문장별 이미지1 확률
        gt = ("img1" if p["qid"][0] % 2 else "img2", "img1" if p["qid"][1] % 2 else "img2")
        pred = ("img1" if s1 > 0.5 else "img2", "img1" if s2 > 0.5 else "img2")
        out.append(dict(p_img1=[s1, s2], pred=list(pred), gt=list(gt), correct=bool(pred == gt),
                        cos=(te @ ie.t()).float().cpu().numpy().round(5).tolist()))
    return out


def summarize(res, pairs):
    per = []
    for c, name in enumerate(CATS):
        k = sum(r["correct"] for r in res[c * 15:(c + 1) * 15])
        per.append(dict(type=name, k=k, acc=100 * k / 15, wilson95=wilson(k, 15)))
    K = sum(x["k"] for x in per)
    margins = [min(abs(r["p_img1"][0] - .5), abs(r["p_img1"][1] - .5)) for r in res]
    near = {str(t): int(sum(m < t for m in margins)) for t in THRS}
    return dict(per_pattern=per, avg=float(np.mean([x["acc"] for x in per])), k_total=K,
                overall_wilson95=wilson(K, 135), near_half_pairs=near, min_margin=float(min(margins)),
                median_margin=float(np.median(margins)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/MMVP_VLM")
    ap.add_argument("--out", default="mmvp_vlm_clip_results.json")
    a = ap.parse_args()
    pairs = load_pairs(a.data)
    assert len(pairs) == 135
    proc = CLIPProcessor.from_pretrained(MODEL)
    configs = [("fp32_cpu", "cpu", torch.float32, "hf"), ("fp32_cpu_openai_pre", "cpu", torch.float32, "openai"),
               ("fp32_mps", "mps", torch.float32, "hf"), ("fp16_mps", "mps", torch.float16, "hf"),
               ("fp32_cpu_rerun", "cpu", torch.float32, "hf"), ("fp16_mps_rerun", "mps", torch.float16, "hf")]  # rerun은 같은 구성의 재실행(결정성 확인). bf16·fp16 CPU는 이 맥에서 10분 넘게 걸려 제외
    results, raw = {}, {}
    for name, dev, dt, pre in configs:
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        try:
            model = CLIPModel.from_pretrained(MODEL, torch_dtype=dt).to(dev).eval()
            res = run(model, proc, pairs, dev, dt, pre)
        except Exception as e:  # 일부 연산이 해당 장치·정밀도에서 지원되지 않을 수 있음
            results[name] = dict(error=repr(e)[:200]); continue
        raw[name] = res
        print(name, "done", flush=True)
        results[name] = summarize(res, pairs)
        results[name].update(device=dev, dtype=str(dt), preprocess=pre)
    # 구성 간 비교
    base = raw["fp32_cpu"]
    for name, res in raw.items():
        flips = [i for i in range(135) if res[i]["correct"] != base[i]["correct"]]
        dec = [(i, j) for i in range(135) for j in range(2) if res[i]["pred"][j] != base[i]["pred"][j]]
        dp = max(abs(res[i]["p_img1"][j] - base[i]["p_img1"][j]) for i in range(135) for j in range(2))
        results[name]["vs_fp32_cpu"] = dict(pair_correct_flips=[dict(pair=i, type=pairs[i]["type"], qid=pairs[i]["qid"],
            base=base[i]["correct"], now=res[i]["correct"]) for i in flips], n_decision_flips=len(dec), max_abs_dp=dp)
    # 쌍 단위 상세(fp32_cpu 기준)
    detail = [dict(pair=i, type=pairs[i]["type"], qid=pairs[i]["qid"], p_img1=[round(x, 5) for x in base[i]["p_img1"]],
                   correct=base[i]["correct"], margin=min(abs(base[i]["p_img1"][0] - .5), abs(base[i]["p_img1"][1] - .5)))
              for i in range(135)]
    # 구성별 쌍 정오와 확률
    per_cfg = {n: [dict(p_img1=[round(x, 5) for x in r["p_img1"]], correct=r["correct"]) for r in res] for n, res in raw.items()}
    # 쌍 하나가 바뀔 때 평균 변화
    json.dump(dict(model=MODEL, table1=TABLE1, one_pair_pattern_points=100 / 15, one_pair_avg_points=100 / 15 / 9,
                   torch=torch.__version__, configs=results, pair_detail_fp32_cpu=detail, per_config_pairs=per_cfg),
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'config':22s} avg   " + " ".join(f"{c[:5]:>5s}" for c in CATS) + "  near<.01/.02/.05")
    for n, r in results.items():
        if "error" in r: print(n, r["error"]); continue
        nh = r["near_half_pairs"]
        print(f"{n:22s} {r['avg']:5.1f} " + " ".join(f"{x['acc']:5.1f}" for x in r["per_pattern"]) +
              f"  {nh['0.01']}/{nh['0.02']}/{nh['0.05']}  flips={len(r['vs_fp32_cpu']['pair_correct_flips'])} dflips={r['vs_fp32_cpu']['n_decision_flips']} max|dp|={r['vs_fp32_cpu']['max_abs_dp']:.2e}")
    print("Table 1               19.3 " + " ".join(f"{x:5.1f}" for x in TABLE1))


if __name__ == "__main__":
    main()
