"""paper/paper.txt(pdftotext -layout)에서 부록 표 12-16, 본문 표 3, 데이터 표 19-20, 표 9, 표 21, 그림 15 상관행렬을 파싱해 tables.json으로 저장.

사용: python3 analysis/parse_tables.py   (저장소의 cambrian1/ 에서)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINES = (ROOT / "paper" / "paper.txt").read_text().split("\n")

COLS = ["avg", "MMEP", "MMB", "SEEDI", "GQA", "SQAI", "MMMU", "MathVista", "AI2D",
        "ChartQA", "OCRBench", "TextVQA", "DocVQA", "MMVP", "RWQA", "CV2D", "CV3D"]
GROUPS = {
    "General": ["MMEP", "MMB", "SEEDI", "GQA"],
    "Knowledge": ["SQAI", "MMMU", "MathVista", "AI2D"],
    "OCR": ["ChartQA", "OCRBench", "TextVQA", "DocVQA"],
    "Vision": ["MMVP", "RWQA", "CV2D", "CV3D"],
}


def num(tok):
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def row_of(line, n=17):
    toks = line.split()
    vals = []
    for t in reversed(toks):
        v = num(t)
        if v is None:
            break
        vals.append(v)
        if len(vals) == n:
            break
    if len(vals) != n:
        return None
    vals.reverse()
    name = " ".join(toks[: len(toks) - n])
    return name, vals


def caption_idx(label):
    for i, l in enumerate(LINES):
        if re.match(r"\s*Table %s \|" % label, l):
            return i
    raise KeyError(label)


def rows_before(label, max_back=45):
    c = caption_idx(label)
    out = []
    for i in range(c - 1, max(c - max_back, 0), -1):
        if re.match(r"\s*Table \d+ \|", LINES[i]):
            break
        r = row_of(LINES[i])
        if r:
            out.append((i + 1, r[0], r[1]))
    out.reverse()
    return out


def rows_after(label, max_fwd=30):
    # 표 12는 캡션 앞에 행이 있다. 표 15 등도 마찬가지이므로 rows_before만 쓴다.
    raise NotImplementedError


def to_dict(rows):
    res = []
    for ln, name, vals in rows:
        d = dict(zip(COLS, vals))
        d["name"] = name
        d["line"] = ln
        res.append(d)
    return res


tables = {}
for lab, key in [("3", "t3_ensemble"), ("12", "t12_0M"), ("13", "t13_0.5M"), ("14", "t14_1.2M"),
                 ("15", "t15_1.2M_unfrozen"), ("16", "t16_5M"), ("19", "t19_balance"),
                 ("20", "t20_ratio")]:
    tables[key] = to_dict(rows_before(lab))

# 표 20 이름 보정(exp1..exp5는 행 이름에 포함됨), 표 19 이름은 't = 150k 4015k' 형태
# 표 12-14 이름의 '구분' 줄(Language-Supervised 등)은 숫자가 없어 자동으로 걸러짐.

# 표 2, 9, 21, 그림 15
t9 = []
for i, l in enumerate(LINES, 1):
    if 1000 < i < 1200 and re.search(r"Cambrian-1-(8|13|34)B|Mini-Gemini-HD-(8|13|34)B|LLaVA-NeXT-(8|13|34)B", l):
        toks = l.split()
        vals = [num(t) for t in toks[1:] if num(t) is not None]
        t9.append({"name": toks[0], "vals": vals, "line": i})
tables["t9_final"] = t9

# 표 21: 이름, 이미지 수, data eng, cambrian10M, llava665k
t21 = []
for i, l in enumerate(LINES, 1):
    if 2500 < i < 2560:
        m = re.search(r"(MMEP|MMB|SEEDI|GQA|SQAI|MMMUV|MathVistaM|AI2D|ChartQA|OCRBench|TextVQA|DocVQA|MMVP|RealWorldQA|CV-Bench)\s+([\d,]+)\s+(\d+) \(([\d.]+)%\)\s+([\d,]+) \(([\d.]+)%\)\s+([\d,]+)\s+\(([\d.]+)%\)", l)
        if m:
            g = m.groups()
            t21.append({"bench": g[0], "n": int(g[1].replace(",", "")), "data_eng": int(g[2]),
                        "cambrian10M": int(g[4].replace(",", "")), "llava665k": int(g[6].replace(",", ""))})
tables["t21_leak"] = t21

# 그림 15 상관행렬
names = ["MME", "MMB", "SEED-I", "GQA", "SQA-I", "MMMU", "MathVista", "AI2D", "ChartQA", "OCRBench",
         "TextVQA", "DocVQA", "MMVP", "RW QA", "CV-CB"]
corr = []
for i, l in enumerate(LINES, 1):
    if 1700 < i < 1760:
        toks = l.split()
        for nm in names:
            pre = nm.split()
            if toks[: len(pre)] == pre:
                vals = [num(t) for t in toks[len(pre):] if num(t) is not None]
                if len(vals) == 15:
                    corr.append((nm, vals))
tables["fig15_names"] = [c[0] for c in corr]
tables["fig15_corr"] = [c[1] for c in corr]

# 표 2 순위 (Language-supervised, Self-supervised): 직접 입력 대신 원문에서 읽은 값을 그대로 옮김(표 2, 10쪽)
tables["t2_ranks"] = {
    "lang": [["SigLIP ViT-SO400M/14@384", 1, 1, 1, 2, 1], ["OpenCLIP ConvNeXt-XXL@1024", 2, 6, 8, 1, 3],
             ["DFN-CLIP ViT-H/14@378", 3, 4, 2, 5, 4], ["OpenCLIP ConvNeXt-L@1024", 4, 8, 7, 3, 8],
             ["SigLIP ViT-L/16@384", 5, 5, 4, 4, 6], ["OpenAI CLIP ViT-L/14@336", 6, 3, 6, 6, 7],
             ["EVA-CLIP-02 ViT-L/14@336", 7, 2, 5, 8, 2], ["OpenCLIP ConvNeXt-L@512", 8, 7, 3, 7, 9],
             ["DFN-CLIP ViT-L/14@224", 9, 9, 9, 9, 10], ["DINOv2* ViT-L/14@518", 10, 10, 10, 10, 5]],
    "ssl": [["DINOv2 ViT-L/14@518", 1, 1, 1, 1, 1], ["DINOv2 ViT-L/14@336", 2, 2, 3, 3, 2],
            ["MAE ViT-L/16@224", 3, 5, 2, 2, 4], ["I-JEPA ViT-H/14@224", 4, 3, 6, 8, 3],
            ["SD2.1 VAE+UNet/16@512", 5, 7, 9, 9, 5], ["MiDaS 3.0 ViT-L/16@384", 6, 6, 8, 5, 6],
            ["SupViT ViT-L/16@224", 7, 4, 9, 4, 8], ["MoCo v3 ViT-B/16@224", 8, 8, 4, 7, 7],
            ["MoCo v3 ViT-L/16@224", 9, 9, 5, 6, 9], ["SAM ViT-H/16@1024", 10, 10, 10, 10, 10]],
    "cols": ["All", "G", "K", "O", "V"],
}

# 본문 표 4-8 (텍스트에서 옮김)
tables["t4_connector"] = {"Concat": [67.2, 48.9, 50.1, 52.6], "Resampler": [63.1, 46.5, 27.1, 42.6],
                          "SVA-no-multi-agg": [68.0, 49.5, 55.2, 52.6], "SVA": [68.5, 49.7, 55.5, 53.2],
                          "cols": ["General", "Knowledge", "OCR", "Vision"]}
tables["t5_compress"] = {"Interpolate+MLP": [63.4, 43.8, 28.1, 43.7], "C-Abstractor": [64.4, 42.8, 26.1, 44.3],
                         "LDPv2": [62.5, 43.9, 28.7, 43.9], "SVA": [65.5, 44.5, 31.4, 46.9],
                         "cols": ["General", "Knowledge", "OCR", "Vision"]}
tables["t6_sva_abl"] = {"D": {"2": 52.1, "3": 52.4, "4": 52.8}, "G": {"1": 52.4, "2": 52.6, "3": 53.1},
                        "multi_agg": {"No": 52.4, "Yes": 53.3}}
tables["t7_balance"] = {"150k": [53.7, 68.0, 51.3, 45.2, 50.5], "250k": [54.3, 68.1, 51.5, 45.3, 52.2],
                        "350k": [54.3, 67.4, 51.4, 46.0, 52.3], "450k": [54.2, 68.0, 52.2, 45.5, 50.7],
                        "cols": ["Average", "General", "Knowledge", "OCR", "Vision"]}
tables["t8_data"] = {"LLaVA-665K": [40.7, 64.7, 45.2, 20.8, 32.0], "Cambrian-10M": [54.8, 68.7, 51.6, 47.3, 51.4],
                     "Cambrian-7M": [55.9, 69.6, 52.6, 47.3, 54.1],
                     "cols": ["Average", "General", "Knowledge", "OCR", "Vision"]}
tables["t24_fuzzy"] = {
    "cols": ["SEEDI", "GQA", "SQAI", "MMMU", "AI2D", "ChartQA", "OCRBench", "TextVQA", "MMVP", "RWQA"],
    "8B_fuzzy": [74.7, 64.6, 80.4, 42.7, 73.0, 73.3, 62.4, 71.7, 51.3, 64.2],
    "8B_gpt35": [78.4, 65.8, 82.0, 38.9, 78.1, 71.2, 67.0, 69.2, 49.3, 63.5],
    "13B_fuzzy": [74.7, 64.6, 80.4, 40.4, 73.0, 73.3, 62.4, 71.7, 51.3, 64.2],
    "13B_gpt35": [77.3, 64.7, 81.3, 37.2, 78.2, 71.4, 67.1, 75.6, 46.0, 64.3],
}
tables["cols"] = COLS
tables["groups"] = GROUPS

(ROOT / "analysis" / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=1))
for k in ["t3_ensemble", "t12_0M", "t13_0.5M", "t14_1.2M", "t15_1.2M_unfrozen", "t16_5M", "t19_balance", "t20_ratio",
          "t9_final", "t21_leak", "fig15_corr"]:
    print(k, len(tables[k]))
