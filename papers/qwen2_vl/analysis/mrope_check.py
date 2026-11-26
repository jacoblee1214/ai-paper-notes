"""M-RoPE와 ViT 2D-RoPE 검산 토이 (PyTorch, CPU, 수 초).

공식 코드를 실행하지 않는다. transformers 소스(`repo/transformers/modeling_qwen2_vl.py`,
`vision_utils.py`, `repo/transformers_v4.45.0/modeling_qwen2_vl.py`)를 정적으로 읽고,
같은 규칙을 직접 다시 구현해 식과 작은 예로 확인한다.

실행: python3 analysis/mrope_check.py   (torch 필요. 결과: analysis/mrope_results.json)
"""
import json
import math
from pathlib import Path

import torch

OUT = Path(__file__).with_name("mrope_results.json")
torch.set_default_dtype(torch.float64)
R = {}


# ----------------------------------------------------------------------------
# 1. 위치 ID (get_rope_index 재구현)
# ----------------------------------------------------------------------------
def rope_index(segments, variant):
    """segments: [("text", n) | ("image"|"video", (t, h, w))], (t, h, w)는 ViT 패치 격자(병합 전).
    variant="v445": transformers v4.45.0 규칙. 다음 구간 시작 = 지금까지 최대 ID + 1.
    variant="main": 2026-11 main 규칙. 비전 구간 뒤 시작 = 시작 + max(h/2, w/2).
    반환: (3, L) 위치 ID, 구간별 (이름, 시작 위치, 끝 위치(포함))."""
    ids, info, cur = [], [], 0
    for kind, spec in segments:
        if kind == "text":
            p = torch.arange(spec).view(1, -1).expand(3, -1) + cur
            ids.append(p)
            info.append((kind, cur, cur + spec - 1))
            cur += spec
        else:
            t, h, w = spec
            lt, lh, lw = t, h // 2, w // 2
            tt = torch.arange(lt).view(-1, 1, 1).expand(lt, lh, lw).flatten() + cur
            hh = torch.arange(lh).view(1, -1, 1).expand(lt, lh, lw).flatten() + cur
            ww = torch.arange(lw).view(1, 1, -1).expand(lt, lh, lw).flatten() + cur
            p = torch.stack([tt, hh, ww])
            ids.append(p)
            info.append((kind, cur, int(p.max())))
            if variant == "v445":
                cur = int(p.max()) + 1
            else:
                cur += max(lh, lw)
    return torch.cat(ids, 1), info


def show(name, segments):
    out = {}
    for v in ("v445", "main"):
        p, info = rope_index(segments, v)
        out[v] = {"t": p[0].tolist(), "h": p[1].tolist(), "w": p[2].tolist(), "segments": info}
    out["same"] = out["v445"]["t"] == out["main"]["t"] and out["v445"]["h"] == out["main"]["h"] and out["v445"]["w"] == out["main"]["w"]
    R[name] = out
    return out


# (a) 텍스트만: 세 성분이 같고 0,1,2,...
a = show("a_text_only", [("text", 5)])
assert a["v445"]["t"] == a["v445"]["h"] == a["v445"]["w"] == list(range(5))

# (b) 텍스트 3 + 이미지(1,4,6 -> 병합 후 2x3) + 텍스트 5. 다음 텍스트는 최대 ID + 1에서 이어진다.
b = show("b_text_image_text", [("text", 3), ("image", (1, 4, 6)), ("text", 5)])
img = b["v445"]["segments"][1]
nxt = b["v445"]["segments"][2]
assert nxt[1] == img[2] + 1, "이미지 뒤 텍스트 = 이미지 최대 ID + 1"
assert b["same"], "이미지에서는 두 규칙이 같다"
R["b_text_image_text"]["check"] = {"image_max_id": img[2], "next_text_start": nxt[1]}

# (c) 영상(3,4,4 -> 병합 후 t=3, 2x2) = 코드 주석의 예. 시간 ID는 시간 패치(프레임 2장)마다 1 증가
c = show("c_video_small", [("video", (3, 4, 4)), ("text", 3)])
R["c_video_small"]["note"] = "t=3 > max(h',w')=2: v445는 텍스트가 3에서, main은 2에서 시작"

# (d) 영상 시간 격자가 공간 격자보다 긴 경우: 60초 x 2fps = 120프레임 -> t'=60, 448x448 -> 16x16
d = show("d_video_long", [("text", 4), ("video", (60, 32, 32)), ("text", 4)])
vseg = d["main"]["segments"][1]
R["d_video_long"]["collision"] = {
    "video_temporal_id_range": [vseg[1], int(max(d["main"]["t"][4:4 + 60 * 16 * 16]))],
    "text_after_start_v445": d["v445"]["segments"][2][1],
    "text_after_start_main": d["main"]["segments"][2][1],
    "n_video_frames_whose_temporal_id_ge_main_text_start": int(sum(1 for t in range(60) if t + vseg[1] >= d["main"]["segments"][2][1])),
}
# 위치 ID 폭: 1D라면 시각 토큰 수만큼 올라가지만 M-RoPE는 max(t', h', w')(v445) 만큼
n_tokens = 60 * 16 * 16
R["d_video_long"]["span"] = {"visual_tokens": n_tokens, "mrope_span_v445": d["v445"]["segments"][2][1] - 4, "mrope_span_main": d["main"]["segments"][2][1] - 4}

# (e) 이미지 한 장: 1D는 h'*w' 칸, M-RoPE는 max(h',w') 칸
e = {}
for name, (h, w) in {"224x224": (16, 16), "1344x1344(제한 없음)": (96, 96), "문서 1240x1754 -> 1232x1750": (88, 125)}.items():
    lh, lw = h // 2, w // 2
    e[name] = {"visual_tokens": lh * lw, "mrope_span": max(lh, lw), "ratio": lh * lw / max(lh, lw)}
R["e_position_span_image"] = e

# ----------------------------------------------------------------------------
# 2. 회전: M-RoPE vs 1D RoPE (LM 쪽, head_dim=128, theta=1e6, section=[16,24,24])
# ----------------------------------------------------------------------------
D, THETA, SECTION = 128, 1_000_000.0, [16, 24, 24]
inv = 1.0 / (THETA ** (torch.arange(0, D, 2).double() / D))  # (64,)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


def cos_sin_1d(pos):  # pos: (L,)
    f = pos[:, None].double() * inv[None]
    f = torch.cat([f, f], -1)
    return f.cos(), f.sin()


def cos_sin_mrope(pos3):  # pos3: (3, L)
    f = pos3[..., None].double() * inv  # (3, L, 64)
    parts = f.split(SECTION, dim=-1)
    f = torch.cat([m[i % 3] for i, m in enumerate(parts)], -1)  # 코드 recomposition_frequencies와 같은 규칙
    f = torch.cat([f, f], -1)
    return f.cos(), f.sin()


def apply(x, cs):
    c, s = cs
    return x * c + rotate_half(x) * s


# (a) 채널 -> 축 대응
axis_of_pair = []
for i, n in enumerate(SECTION):
    axis_of_pair += [i % 3] * n
wl = (2 * math.pi / inv).tolist()
bands = {}
for ax, name in enumerate(["temporal", "height", "width"]):
    idx = [k for k, a_ in enumerate(axis_of_pair) if a_ == ax]
    bands[name] = {"pairs": len(idx), "first": idx[0], "last": idx[-1], "share": len(idx) / 64,
                   "wavelength_min": wl[idx[0]], "wavelength_max": wl[idx[-1]]}
R["channel_split"] = bands

# (b) 텍스트: 세 성분이 같으면 1D RoPE와 비트 단위로 같다
L = 50
pos = torch.arange(L)
c1 = cos_sin_1d(pos)
c3 = cos_sin_mrope(pos.view(1, -1).expand(3, -1))
R["text_equals_1d"] = {"max_abs_diff_cos": float((c1[0] - c3[0]).abs().max()), "max_abs_diff_sin": float((c1[1] - c3[1]).abs().max())}
assert R["text_equals_1d"]["max_abs_diff_cos"] == 0.0

# (c) 상대 위치 성질: 점곱이 축별 상대 위치만의 함수 (두 토큰을 같은 양만큼 평행 이동해도 점수 불변)
g = torch.Generator().manual_seed(0)
q = torch.randn(D, generator=g)
k = torch.randn(D, generator=g)
errs = []
for _ in range(200):
    pq = torch.randint(0, 40, (3, 1), generator=g)
    pk = torch.randint(0, 40, (3, 1), generator=g)
    sh = torch.randint(0, 5000, (3, 1), generator=g)
    s0 = (apply(q[None], cos_sin_mrope(pq)) * apply(k[None], cos_sin_mrope(pk))).sum()
    s1 = (apply(q[None], cos_sin_mrope(pq + sh)) * apply(k[None], cos_sin_mrope(pk + sh))).sum()
    errs.append(float((s0 - s1).abs() / (s0.abs() + 1e-12)))
R["shift_invariance_mrope"] = {"trials": 200, "max_rel_err": max(errs)}

# 높이 방향 이동과 너비 방향 이동은 서로 다른 채널을 쓴다: 한 축만 이동하면 다른 축 채널 점수는 변하지 않는다
pq = torch.tensor([[0], [3], [5]])
pk_h = torch.tensor([[0], [7], [5]])  # 높이만 +4
pk_w = torch.tensor([[0], [3], [9]])  # 너비만 +4


def per_axis_score(pq, pk):
    qq, kk = apply(q[None], cos_sin_mrope(pq)), apply(k[None], cos_sin_mrope(pk))
    prod = (qq * kk)[0]
    # rotate_half 배치: 채널 i와 i+64가 한 쌍. 축별 합을 구한다
    pair = prod[:64] + prod[64:]
    out = {}
    for ax, name in enumerate(["temporal", "height", "width"]):
        out[name] = float(sum(pair[j] for j, a_ in enumerate(axis_of_pair) if a_ == ax))
    return out


base = per_axis_score(pq, pq)
R["axis_separation"] = {"same_position": base, "height_plus4": per_axis_score(pq, pk_h), "width_plus4": per_axis_score(pq, pk_w)}
assert abs(R["axis_separation"]["height_plus4"]["width"] - base["width"]) < 1e-9
assert abs(R["axis_separation"]["width_plus4"]["height"] - base["height"]) < 1e-9

# (d) 영상의 시간 IDs가 늘어날 때: 같은 위치(h,w) 두 프레임 사이 점수는 시간 채널(앞 16쌍, 가장 빠른 주파수)만 바뀐다
# 주파수 대역 요약: 시간 축 파장 6.3~159, 높이 축 200~약 4,500(코드 section 순서), 너비 축은 가장 느린 채널
R["note_frequency"] = "temporal=가장 빠른 16쌍, height=다음 24쌍, width=가장 느린 24쌍 (section=[16,24,24], 코드 chunk 순서)"


# (e) 격자 폭 S 안에서 한 칸 이동이 만드는 회전각: 축별로 각이 1 rad 이상 도는 채널 쌍 수
# 높이·너비 위치는 0..S-1 범위(병합 후 격자, 1120 px 이미지면 S=40, 1792 px이면 S=64)
def resolvable(axis_of, span, thr=1.0):
    out = {}
    for ax, name in enumerate(["temporal", "height", "width"]):
        idx = [j for j, a_ in enumerate(axis_of) if a_ == ax]
        ang = [float(inv[j]) * (span - 1) for j in idx]
        out[name] = {"pairs": len(idx), "pairs_over_1rad": sum(a_ >= thr for a_ in ang), "max_angle_rad": max(ang), "min_angle_rad": min(ang)}
    return out


axis_qwen3 = [0] * 64  # Qwen3-VL 인터리브: section=[24,20,20], H=idx 1,4,7.., W=idx 2,5,8.. (modeling_qwen3_vl.py recomposition_frequencies)
for j in range(1, 60, 3):
    axis_qwen3[j] = 1
for j in range(2, 60, 3):
    axis_qwen3[j] = 2
R["resolvable_pairs"] = {f"span{S}": {"qwen2_vl_chunked": resolvable(axis_of_pair, S), "qwen3_vl_interleaved(theta 1e6로 가정)": resolvable(axis_qwen3, S)} for S in (16, 40, 64)}

# ----------------------------------------------------------------------------
# 3. ViT 2D RoPE (head_dim=80, theta=1e4): 블록 순서와 평행 이동 불변성
# ----------------------------------------------------------------------------
def vit_positions(h, w, merge=2):
    hp, wp = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    return hp.reshape(shape).transpose(1, 2).flatten(), wp.reshape(shape).transpose(1, 2).flatten()


hp, wp = vit_positions(4, 4)
R["vit_block_order_4x4"] = {"h": hp.tolist(), "w": wp.tolist(),
                            "first_four": list(zip(hp[:4].tolist(), wp[:4].tolist()))}
assert list(zip(hp[:4].tolist(), wp[:4].tolist())) == [(0, 0), (0, 1), (1, 0), (1, 1)], "연속 4개가 2x2 블록"

HD = 80
sp = HD // 2
vinv = 1.0 / (10000.0 ** (torch.arange(0, sp, 2).double() / sp))  # 20개


def vit_cos_sin(h_pos, w_pos):
    fh, fw = h_pos[:, None].double() * vinv, w_pos[:, None].double() * vinv
    f = torch.cat([fh, fw], -1)  # (N, 40)
    f = torch.cat([f, f], -1)    # (N, 80)
    return f.cos(), f.sin()


qv, kv = torch.randn(HD, generator=g), torch.randn(HD, generator=g)
errs = []
for _ in range(200):
    a_ = torch.randint(0, 60, (2, 1), generator=g)
    b_ = torch.randint(0, 60, (2, 1), generator=g)
    s_ = torch.randint(0, 500, (2, 1), generator=g)
    s0 = (apply(qv[None], vit_cos_sin(a_[0], a_[1])) * apply(kv[None], vit_cos_sin(b_[0], b_[1]))).sum()
    s1 = (apply(qv[None], vit_cos_sin(a_[0] + s_[0], a_[1] + s_[1])) * apply(kv[None], vit_cos_sin(b_[0] + s_[0], b_[1] + s_[1]))).sum()
    errs.append(float((s0 - s1).abs() / (s0.abs() + 1e-12)))
R["vit_shift_invariance"] = {"trials": 200, "max_rel_err": max(errs), "freqs_per_axis": int(vinv.numel())}

# 절대 위치 임베딩 대비: ViT 2D-RoPE는 격자 크기와 무관하게 같은 가중치를 쓴다 (격자 크기가 달라도 같은 식)

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))
print(json.dumps({k: R[k] for k in ("channel_split", "text_equals_1d", "shift_invariance_mrope", "vit_shift_invariance")}, indent=1, ensure_ascii=False))
print("(c) v445:", c := R["c_video_small"]["v445"]["t"], R["c_video_small"]["v445"]["segments"])
print("(c) main:", R["c_video_small"]["main"]["t"], R["c_video_small"]["main"]["segments"])
print("(d) collision:", R["d_video_long"]["collision"], R["d_video_long"]["span"])
print("(e) span:", R["e_position_span_image"])
