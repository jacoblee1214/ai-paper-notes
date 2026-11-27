"""get_rope_index 실측 (D6 확인): 같은 입력을 네 구현에 넣어 위치 번호를 원소별로 비교한다.

가중치는 쓰지 않는다(모델 가중치 다운로드 없음, 공식 코드 학습 없음, API 호출 없음).
호출 방식:
  installed : 설치된 transformers 패키지의 Qwen2VLModel을 아주 작은 설정으로 만들어 get_rope_index를 그대로 호출
  main      : repo/transformers/modeling_qwen2_vl.py에서 두 메서드(get_vision_position_ids, get_rope_index)의
              소스를 ast로 뽑아 exec, 설정만 흉내 낸 stub을 self로 넘김 (의존성 검사를 피하려는 정적 추출)
  v445      : repo/transformers_v4.45.0/modeling_qwen2_vl.py의 get_rope_index를 같은 방식으로 추출·실행
  official  : repo/Qwen2-VL/qwen-vl-finetune/qwenvl/data/rope2d.py의 get_rope_index_2를 importlib로 불러 실행
같은 파일이면 해시를 기록한다.

실행: HF_HUB_OFFLINE=1 python3 analysis/rope_index_check.py [--main PATH] [--v445 PATH] [--official PATH]
      (torch, transformers가 설치된 파이썬. 결과: analysis/rope_index_results.json)
"""
import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import textwrap
import types
from pathlib import Path
from typing import Optional, Tuple

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "repo"
ap = argparse.ArgumentParser()
ap.add_argument("--main", default=str(REPO / "transformers" / "modeling_qwen2_vl.py"))
ap.add_argument("--v445", default=str(REPO / "transformers_v4.45.0" / "modeling_qwen2_vl.py"))
ap.add_argument("--official", default=str(REPO / "Qwen2-VL" / "qwen-vl-finetune" / "qwenvl" / "data" / "rope2d.py"))
ap.add_argument("--out", default=str(HERE / "rope_index_results.json"))
args = ap.parse_args()

IMG, VID, VSTART, VEND, PAD = 151655, 151656, 151652, 151653, 151643
MERGE = 2


# ----------------------------------------------------------------------------
# 구현 불러오기
# ----------------------------------------------------------------------------
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def extract_methods(path, names):
    """modeling 파일에서 지정한 메서드들의 소스를 뽑아 한 네임스페이스에 정의한다."""
    src = Path(path).read_text(encoding="utf-8")
    found = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = textwrap.dedent(ast.get_source_segment(src, node))
    ns = {"torch": torch, "itertools": itertools, "Optional": Optional, "Tuple": Tuple}
    exec("from __future__ import annotations\n", ns)
    for n in names:
        if n in found:
            exec(found[n], ns)
    return ns, sorted(found)


def make_stub(ns, with_ids):
    cfg = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=MERGE),
        image_token_id=IMG, video_token_id=VID, vision_start_token_id=VSTART,
    )
    stub = types.SimpleNamespace(config=cfg)
    if "get_vision_position_ids" in ns:
        stub.get_vision_position_ids = types.MethodType(ns["get_vision_position_ids"], stub)
    return stub


def load_modeling_impl(path):
    ns, found = extract_methods(path, ["get_rope_index", "get_vision_position_ids"])
    stub = make_stub(ns, True)
    fn = types.MethodType(ns["get_rope_index"], stub)
    return fn, found


def load_official(path):
    spec = importlib.util.spec_from_file_location("official_rope2d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_rope_index_2


def load_installed():
    import transformers
    from transformers import Qwen2VLConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel
    cfg = Qwen2VLConfig(
        text_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, vocab_size=151936, max_position_embeddings=64,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, embed_dim=16, hidden_size=16, num_heads=2, spatial_merge_size=MERGE),
        image_token_id=IMG, video_token_id=VID, vision_start_token_id=VSTART, vision_end_token_id=VEND,
    )
    model = Qwen2VLModel(cfg).eval()
    return model, transformers.__version__


# ----------------------------------------------------------------------------
# 입력 만들기: segments = [("text", n) | ("image"|"video", (t, h, w))], 격자는 병합 전
# 비전 구간은 <vision_start> 토큰 N개 <vision_end> 로 감싼다 (start/end는 텍스트 취급)
# ----------------------------------------------------------------------------
def build(segments):
    ids, kinds, vis = [], [], []  # kinds: 0 텍스트, 1 이미지, 2 영상 (mm_token_type_ids)
    img_grids, vid_grids = [], []
    for kind, spec in segments:
        if kind == "text":
            ids += [1000 + (len(ids) + i) % 500 for i in range(spec)]
            kinds += [0] * spec
        else:
            t, h, w = spec
            n = t * (h // MERGE) * (w // MERGE)
            ids.append(VSTART); kinds.append(0)
            s = len(ids)
            ids += [IMG if kind == "image" else VID] * n
            kinds += [1 if kind == "image" else 2] * n
            vis.append((kind, s, s + n))
            ids.append(VEND); kinds.append(0)
            (img_grids if kind == "image" else vid_grids).append(list(spec))
    return ids, kinds, vis, img_grids, vid_grids


def toy_positions(segments, variant):
    """analysis/mrope_check.py의 규칙(재구현)과 대조하기 위한 같은 규칙의 최소 구현.
    start/end 토큰은 인접 텍스트 구간에 합쳐 센다."""
    ids, cur = [], 0
    out = []
    for kind, spec in segments:
        if kind == "text":
            out.append(["text", spec])
        else:
            out.append([kind, spec])
    # start/end 토큰 포함: 비전 구간 앞 텍스트 +1, 뒤 텍스트 +1
    merged = []
    for kind, spec in out:
        if kind != "text":
            if merged and merged[-1][0] == "text":
                merged[-1][1] += 1
            else:
                merged.append(["text", 1])
            merged.append([kind, spec])
            merged.append(["text", 1])
        else:
            if merged and merged[-1][0] == "text":
                merged[-1][1] += spec
            else:
                merged.append(["text", spec])
    P = []
    for kind, spec in merged:
        if kind == "text":
            P.append(torch.arange(spec).view(1, -1).expand(3, -1) + cur)
            cur += spec
        else:
            t, h, w = spec
            lt, lh, lw = t, h // MERGE, w // MERGE
            tt = torch.arange(lt).view(-1, 1, 1).expand(lt, lh, lw).flatten() + cur
            hh = torch.arange(lh).view(1, -1, 1).expand(lt, lh, lw).flatten() + cur
            ww = torch.arange(lw).view(1, 1, -1).expand(lt, lh, lw).flatten() + cur
            p = torch.stack([tt, hh, ww])
            P.append(p)
            cur = int(p.max()) + 1 if variant == "v445" else cur + max(lh, lw)
    return torch.cat(P, 1)


CASES = {
    "a_text_only": [("text", 5)],
    "b_text_image_text": [("text", 3), ("image", (1, 4, 6)), ("text", 5)],
    "c_video_small_t3": [("text", 2), ("video", (3, 4, 4)), ("text", 3)],
    "c2_video_t_eq_max": [("text", 2), ("video", (2, 4, 4)), ("text", 3)],
    "c3_video_t_lt_max": [("text", 2), ("video", (2, 8, 8)), ("text", 3)],
    "d_video_long_t60": [("text", 4), ("video", (60, 32, 32)), ("text", 4)],
    "f_image_video_mix": [("text", 3), ("image", (1, 8, 8)), ("text", 2), ("video", (6, 4, 4)), ("text", 4)],
    "g_two_videos": [("text", 2), ("video", (5, 4, 4)), ("text", 2), ("video", (5, 4, 4)), ("text", 3)],
    "h_video_then_image": [("text", 2), ("video", (6, 4, 4)), ("text", 2), ("image", (1, 4, 4)), ("text", 3)],
    "i_video_no_trailing_text": [("text", 2), ("video", (6, 4, 4))],
}


def run_case(impls, installed, segments, tag, left_pad=0):
    ids, kinds, vis, ig, vg = build(segments)
    L = len(ids)
    input_ids = torch.tensor([ids])
    mm = torch.tensor([kinds], dtype=torch.int32)
    mask = torch.ones_like(input_ids)
    image_grid = torch.tensor(ig) if ig else None
    video_grid = torch.tensor(vg) if vg else None
    outs, errs = {}, {}
    for name, f in impls.items():
        try:
            if name == "official":
                p, d = f(MERGE, input_ids.clone(), image_grid, video_grid, None, mask.clone())
            elif name == "v445":
                p, d = f(input_ids.clone(), image_grid, video_grid, mask.clone())
            else:  # main, installed: mm_token_type_ids 필요
                p, d = f(input_ids.clone(), mm, image_grid, video_grid, mask.clone())
            outs[name] = (p[:, 0].clone(), d.flatten().tolist())
        except Exception as e:  # 오류도 결과로 기록
            errs[name] = f"{type(e).__name__}: {str(e)[:120]}"
    return outs, errs, vis, L


def compare(outs, ref="official"):
    res = {}
    r = outs[ref][0]
    for name, (p, d) in outs.items():
        diff = (p != r)
        any_t = diff.any(0)
        idx = any_t.nonzero().flatten().tolist()
        res[name] = {
            "equal_to_official": bool(not any_t.any()) and d == outs[ref][1],
            "n_diff_tokens": len(idx),
            "diff_axes": {ax: int(diff[i].sum()) for i, ax in enumerate("thw")},
            "first_diff_index": idx[0] if idx else None,
            "last_diff_index": idx[-1] if idx else None,
            "deltas": d,
            "max_abs_diff": int((p - r).abs().max()),
        }
    return res


def overlap_report(p, vis, L):
    """영상 구간마다: 뒤따르는 첫 토큰(vision_end)의 번호가 영상 시간 번호 최댓값 이하인가."""
    rep = []
    for kind, s, e in vis:
        if kind != "video" or e >= L:
            continue
        tmax = int(p[0, s:e].max())
        nxt = int(p[0, e])  # vision_end 토큰의 번호 = 영상 뒤 텍스트의 시작 번호
        last = int(p[0].max())
        rep.append({"video_token_range": [s, e - 1], "video_temporal_id_max": tmax,
                    "next_text_start": nxt, "overlap": nxt <= tmax, "gap_next_minus_tmax": nxt - tmax})
    return rep


def main():
    R = {"files": {}, "calls": {}, "cases": {}}
    impls = {}
    model, tv = load_installed()
    R["calls"]["installed"] = f"transformers {tv}: Qwen2VLModel(작은 설정, 가중치 무작위).get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask)"
    impls["installed"] = model.get_rope_index
    import transformers.models.qwen2_vl.modeling_qwen2_vl as inst_mod
    R["files"]["installed"] = {"version": tv, "sha256_16": sha(inst_mod.__file__)}

    f, found = load_modeling_impl(args.main)
    impls["main"] = f
    R["files"]["main"] = {"sha256_16": sha(args.main), "same_file_as_installed": sha(args.main) == sha(inst_mod.__file__), "methods_extracted": found}
    R["calls"]["main"] = "ast로 메서드 소스를 뽑아 exec, self=stub(config.vision_config.spatial_merge_size=2, 토큰 id 세 개, get_vision_position_ids 바인딩), 인자는 installed와 같음"

    f, found = load_modeling_impl(args.v445)
    impls["v445"] = f
    R["files"]["v445"] = {"sha256_16": sha(args.v445), "methods_extracted": found}
    R["calls"]["v445"] = "ast로 get_rope_index 소스를 뽑아 exec, self=stub(config), 인자: input_ids, image_grid_thw, video_grid_thw, attention_mask"

    impls["official"] = load_official(args.official)
    R["files"]["official"] = {"sha256_16": sha(args.official), "function": "get_rope_index_2"}
    R["calls"]["official"] = "rope2d.py를 importlib로 불러 get_rope_index_2(spatial_merge_size=2, input_ids, image_grid_thw, video_grid_thw, None, attention_mask) 호출"

    for tag, segs in CASES.items():
        outs, errs, vis, L = run_case(impls, model, segs, tag)
        entry = {"segments": [[k, list(s) if isinstance(s, tuple) else s] for k, s in segs], "length": L,
                 "errors": errs}
        if "official" in outs:
            entry["compare_to_official"] = compare(outs)
            entry["overlap_after_video"] = {n: overlap_report(o[0], vis, L) for n, o in outs.items()}
            # 직접 구현(mrope_check.py의 규칙)과 대조
            if any(k != "text" for k, _ in segs):
                tv445, tmain = toy_positions(segs, "v445"), toy_positions(segs, "main")
                entry["toy_rule_matches_measured"] = {
                    "v445_rule==official": bool((tv445 == outs["official"][0]).all()),
                    "v445_rule==v445": bool((tv445 == outs["v445"][0]).all()) if "v445" in outs else None,
                    "main_rule==main": bool((tmain == outs["main"][0]).all()) if "main" in outs else None,
                    "main_rule==installed": bool((tmain == outs["installed"][0]).all()) if "installed" in outs else None,
                }
            # 구간 시작·끝 번호 요약 (가시성)
            entry["positions_summary"] = {n: {"t_max": int(o[0][0].max()), "h_max": int(o[0][1].max()), "w_max": int(o[0][2].max())} for n, o in outs.items()}
            if L <= 80:
                entry["positions"] = {n: {"t": o[0][0].tolist(), "h": o[0][1].tolist(), "w": o[0][2].tolist()} for n, o in outs.items()}
        R["cases"][tag] = entry

    # 패딩(왼쪽) 배치: 행0 = 왼쪽 패딩 3칸 + 영상 케이스, 행1 = 이미지 케이스. 유효 위치만 비교
    segsA = CASES["c_video_small_t3"]
    segsB = CASES["b_text_image_text"]
    idsA, kA, vA, igA, vgA = build(segsA)
    idsB, kB, vB, igB, vgB = build(segsB)
    Lm = max(len(idsA), len(idsB))
    pad = lambda ids, k: ([PAD] * (Lm - len(ids)) + ids, [0] * (Lm - len(ids)) + k, [0] * (Lm - len(ids)) + [1] * len(ids))
    (ia, ka, ma), (ib, kb, mb) = pad(idsA, kA), pad(idsB, kB)
    input_ids = torch.tensor([ia, ib]); mm = torch.tensor([ka, kb], dtype=torch.int32); mask = torch.tensor([ma, mb])
    image_grid = torch.tensor(igB); video_grid = torch.tensor(vgA)
    pad_out = {}
    for name, f in impls.items():
        if name == "official":
            p, d = f(MERGE, input_ids.clone(), image_grid, video_grid, None, mask.clone())
        elif name == "v445":
            p, d = f(input_ids.clone(), image_grid, video_grid, mask.clone())
        else:
            p, d = f(input_ids.clone(), mm, image_grid, video_grid, mask.clone())
        pad_out[name] = (p, d.flatten().tolist())
    ref = pad_out["official"]
    valid = mask.bool()
    pr = {}
    for name, (p, d) in pad_out.items():
        diff = (p != ref[0]) & valid.unsqueeze(0)
        padded_vals = sorted(set(p[:, ~valid].flatten().tolist()))
        pr[name] = {"n_diff_valid_tokens": int(diff.any(0).sum()), "deltas": d, "values_at_padded_positions": padded_vals,
                    "row0_valid_t": p[0, 0, valid[0]].tolist(), "row1_valid_t": p[0, 1, valid[1]].tolist()}
    R["cases"]["j_left_padding_batch"] = {"note": "행0=영상 t3 케이스(왼쪽 패딩), 행1=이미지 케이스. 유효 토큰만 비교", "length": Lm, "per_impl": pr}

    # 요약
    summ = {}
    for name in ("installed", "main", "v445"):
        eq = [t for t, e in R["cases"].items() if "compare_to_official" in e and name in e["compare_to_official"] and e["compare_to_official"][name]["equal_to_official"]]
        ne = [t for t, e in R["cases"].items() if "compare_to_official" in e and name in e["compare_to_official"] and not e["compare_to_official"][name]["equal_to_official"]]
        summ[name] = {"equal_to_official": eq, "differs_from_official": ne}
    R["summary"] = summ
    Path(args.out).write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps({"files": R["files"], "summary": summ}, indent=1, ensure_ascii=False))
    for t, e in R["cases"].items():
        if "compare_to_official" in e:
            print(t, "|", {n: (c["equal_to_official"], c["n_diff_tokens"], c["diff_axes"]) for n, c in e["compare_to_official"].items() if n != "official"}, e.get("errors"))
            print("   overlap:", {n: [(o["video_temporal_id_max"], o["next_text_start"], o["overlap"]) for o in v] for n, v in e["overlap_after_video"].items()})
        elif "errors" in e:
            print(t, e["errors"])
    print("padding:", json.dumps(R["cases"]["j_left_padding_batch"]["per_impl"], ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
