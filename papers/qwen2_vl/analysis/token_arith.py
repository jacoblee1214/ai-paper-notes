"""토큰 수, 파라미터, 계산량 산술 (표준 라이브러리만, 수 초). 공식 코드는 실행하지 않는다.

smart_resize는 `repo/transformers/image_processing_qwen2_vl.py`의 함수를 같은 규칙으로 다시 쓴 것이다.
config 값은 Hugging Face Hub의 config.json(2B/7B/72B, 1 KB)과 `repo/transformers/configuration_qwen2_vl.py` 기본값이다.

실행: python3 analysis/token_arith.py   (결과: analysis/token_results.json)
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("token_results.json")
R = {}


def smart_resize(h, w, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    if max(h, w) / min(h, w) > 200:
        raise ValueError("aspect ratio > 200")
    hb, wb = round(h / factor) * factor, round(w / factor) * factor  # Python round: 0.5는 짝수로
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


CODE_DEFAULT = dict(min_pixels=56 * 56, max_pixels=28 * 28 * 1280)       # transformers 코드 기본 (1280토큰)
HUB_DEFAULT = dict(min_pixels=3136, max_pixels=12845056)                  # Hub preprocessor_config.json (16384토큰)
assert HUB_DEFAULT["max_pixels"] == 16384 * 28 * 28 and HUB_DEFAULT["min_pixels"] == 4 * 28 * 28


def tokens(h, w, cfg):
    hb, wb = smart_resize(h, w, 28, **cfg)
    n = (hb // 28) * (wb // 28)
    return {"resized": [hb, wb], "llm_tokens": n, "vit_patches": 4 * n, "with_vision_start_end": n + 2}


# ---- 1. 경계 규칙 -------------------------------------------------------------
R["rules"] = {
    "round_is_bankers": {"h=42 -> ": smart_resize(42, 1000, 28, **HUB_DEFAULT)[0], "h=70 -> ": smart_resize(70, 1000, 28, **HUB_DEFAULT)[0],
                         "설명": "42/28=1.5와 70/28=2.5는 Python round가 짝수로 보내 56, 56이 된다 (ceil이 아니다)"},
    "h14_zero_then_min_rule": smart_resize(14, 2000, 28, **HUB_DEFAULT),
}
# ceil(H/28)*ceil(W/28) 근사가 맞는 비율 (범위 안에서). 224..1400 정사각이 아닌 모든 (H,W) 중
mism = tot = 0
for hh in range(100, 1000, 7):
    for ww in range(100, 1000, 7):
        a = tokens(hh, ww, HUB_DEFAULT)["llm_tokens"]
        b = math.ceil(hh / 28) * math.ceil(ww / 28)
        tot += 1
        mism += a != b
R["ceil_formula_mismatch"] = {"grid": "H,W in 100..999 step 7", "n": tot, "mismatch": mism, "rate": mism / tot}

# ---- 2. 해상도별 토큰 수 ------------------------------------------------------
resolutions = {
    "224x224 (논문 예, VLA 흔한 값)": (224, 224),
    "256x256 (VLA 흔한 값)": (256, 256),
    "336x336": (336, 336),
    "448x448": (448, 448),
    "672x672": (672, 672),
    "1344x1344": (1344, 1344),
    "1920x1080": (1080, 1920),
    "문서 A4 150dpi 1240x1754": (1754, 1240),
    "문서 A4 300dpi 2480x3508": (3508, 2480),
    "4032x3024 (사진)": (3024, 4032),
}
R["image_tokens"] = {name: {"code_default(max 1280토큰)": tokens(h, w, CODE_DEFAULT), "hub_default(max 16384토큰)": tokens(h, w, HUB_DEFAULT)} for name, (h, w) in resolutions.items()}
# 논문 표 7의 고정 토큰 수가 정사각 격자인지
R["table7_fixed_tokens_square"] = {str(n): {"side_tokens": math.isqrt(n), "perfect_square": math.isqrt(n) ** 2 == n, "pixels": math.isqrt(n) * 28} for n in (64, 576, 1600, 3136)}
# 논문 정의: 224x224 -> 66 (= 64 + 시작/끝 토큰)
assert tokens(224, 224, HUB_DEFAULT)["with_vision_start_end"] == 66

# ---- 3. 영상 -------------------------------------------------------------------
FPS, FRAME_FACTOR = 2.0, 2
SEQ = 128000  # qwen-vl-utils 기본 MODEL_SEQ_LEN
FRAME_MIN_PIX, FRAME_MAX_PIX = 128 * 28 * 28, 768 * 28 * 28  # HEAD 상수(Qwen3-VL 시기, factor는 28로 환산)


def video_tokens(seconds, h, w, fps=FPS, max_frames=768, per_frame_cfg=None):
    n = seconds * fps
    n = min(max(n, 4), max_frames)
    n = int(n // FRAME_FACTOR * FRAME_FACTOR)
    total_pixels = SEQ * 28 * 28 * 0.9
    max_pix = max(min(FRAME_MAX_PIX, total_pixels / n * FRAME_FACTOR), int(FRAME_MIN_PIX * 1.05))
    hb, wb = smart_resize(h, w, 28, min_pixels=FRAME_MIN_PIX, max_pixels=max_pix)
    per_pair = (hb // 28) * (wb // 28)
    return {"frames": n, "temporal_groups": n // 2, "frame_resized": [hb, wb], "tokens_per_group": per_pair, "llm_tokens": (n // 2) * per_pair,
            "tokens_per_second": (n // 2) * per_pair / seconds}


R["video_tokens"] = {
    "10초 640x360": video_tokens(10, 360, 640),
    "60초 640x360": video_tokens(60, 360, 640),
    "60초 448x448 강제": video_tokens(60, 448, 448, per_frame_cfg=None),
    "20분 1280x720 (max 768프레임)": video_tokens(1200, 720, 1280),
}
# 논문 학습 한도: 영상당 토큰 16384. 2fps에서 한 시간당 토큰 수 예산
R["video_budget_16384"] = {"seconds_at_448x448": 16384 / (16 * 16) * 2 / 2, "설명": "448x448(16x16=256 토큰/2프레임)이면 16384토큰 = 64 시간 묶음 = 128프레임 = 64초 (2fps)"}
R["video_budget_16384"]["seconds_at_448x448"] = 16384 // 256 * 2 / 2.0 / 1.0  # 64 묶음 x 2프레임 / 2fps = 64초
# VLA: 같은 224x224 4프레임을 영상 묶음(2프레임당 1격자) 대 이미지 4장
R["vla"] = {
    "image_224_each": tokens(224, 224, HUB_DEFAULT),
    "image_256_each": tokens(256, 256, HUB_DEFAULT),
    "4_frames_as_4_images_224": 4 * tokens(224, 224, HUB_DEFAULT)["llm_tokens"],
    "4_frames_as_video_224": 2 * 64,
    "3_cameras_224": 3 * tokens(224, 224, HUB_DEFAULT)["with_vision_start_end"],
    "3_cameras_x_2_timesteps_as_video_224": 3 * (64 + 2),
}

# ---- 4. 파라미터 ---------------------------------------------------------------
VIS = dict(depth=32, embed=1280, heads=16, mlp=4, patch=14, tpatch=2, in_ch=3, merge=2)


def vit_params(out_dim):
    e = VIS["embed"]
    patch = VIS["in_ch"] * VIS["tpatch"] * VIS["patch"] ** 2 * e  # Conv3d, bias 없음
    blk = (e * 3 * e + 3 * e) + (e * e + e) + 2 * (2 * e) + (e * VIS["mlp"] * e + VIS["mlp"] * e) + (VIS["mlp"] * e * e + e)
    h4 = e * VIS["merge"] ** 2
    merger = 2 * e + (h4 * h4 + h4) + (h4 * out_dim + out_dim)
    return {"patch_embed": patch, "blocks_total": VIS["depth"] * blk, "per_block": blk, "merger": merger,
            "vit_no_merger": patch + VIS["depth"] * blk, "vit_with_merger": patch + VIS["depth"] * blk + merger}


def lm_params(c):
    h, i, L, V = c["h"], c["i"], c["L"], c["V"]
    kv = c["kv"] * (h // c["heads"])
    attn = (h * h + h) + (h * kv + kv) * 2 + h * h  # q,k,v bias 있음, o 없음
    mlp = 3 * h * i
    layer = attn + mlp + 2 * h
    emb = V * h
    head = 0 if c["tied"] else V * h
    return {"embed": emb, "lm_head": head, "layers": L * layer, "per_layer": layer, "final_norm": h,
            "total": emb + head + L * layer + h, "non_embedding": L * layer + h}


CFG = {
    "2B": dict(h=1536, i=8960, L=28, V=151936, kv=2, heads=12, tied=True, reported_lm=1.5e9, hub_total=2208985600),
    "7B": dict(h=3584, i=18944, L=28, V=152064, kv=4, heads=28, tied=False, reported_lm=7.6e9, hub_total=8291375616),
    "72B": dict(h=8192, i=29568, L=80, V=152064, kv=8, heads=64, tied=False, reported_lm=72e9, hub_total=73405560320),
}
P = {}
for name, c in CFG.items():
    v, l = vit_params(c["h"]), lm_params(c)
    tot = v["vit_with_merger"] + l["total"]
    P[name] = {"vit": v, "lm": l, "total_calc": tot, "hub_safetensors_total": c["hub_total"], "diff_vs_hub": tot - c["hub_total"],
               "reported_vit": 675e6, "reported_lm": c["reported_lm"]}
R["params"] = P

# ---- 5. 계산량 (순전파 FLOPs, 곱셈-덧셈=2 FLOPs) ------------------------------
def vit_flops(h_px, w_px, frames_groups=1):
    """ViT 한 번 통과. 프레임 하나(2프레임 묶음 하나)가 한 attention 구간이다 (cu_seqlens = h*w per t)."""
    hb, wb = smart_resize(h_px, w_px, 28, **HUB_DEFAULT)
    n = (hb // 14) * (wb // 14)  # 병합 전 패치 수
    e = VIS["embed"]
    lin = 2 * n * (VIS["depth"] * (e * 3 * e + e * e + 2 * e * 4 * e) + VIS["tpatch"] * 0 + VIS["in_ch"] * VIS["tpatch"] * 196 * e)
    att = VIS["depth"] * 4 * n * n * e  # QK^T 와 AV 각각 2 n^2 e
    merger = 2 * (n // 4) * ((4 * e) ** 2)  # 첫 선형만 (둘째는 LM 폭에 따라)
    return {"patches": n, "linear_flops": lin * frames_groups, "attn_flops": att * frames_groups, "total": (lin + att + merger) * frames_groups,
            "attn_share": att / (lin + att + merger)}


R["vit_flops"] = {name: vit_flops(h, w) for name, (h, w) in {"224": (224, 224), "256": (256, 256), "448": (448, 448), "1344(max 16384토큰)": (1344, 1344), "1344 -> max 1280토큰 제한": (980, 980),
                                                              "문서 1240x1754": (1754, 1240)}.items()}
# LM 비임베딩 FLOPs/토큰 = 2 x non_embedding
R["lm_flops_per_token"] = {k: 2 * v["lm"]["non_embedding"] for k, v in P.items()}
# 시각 토큰 1개당 ViT 비용 (패치 4개를 거침) 대 LM 비용
R["vit_per_visual_token_vs_lm"] = {}
base = vit_flops(448, 448)
per_token = base["total"] / (base["patches"] // 4)
for k in P:
    R["vit_per_visual_token_vs_lm"][k] = {"vit_flops_per_llm_visual_token(448x448)": per_token, "lm_flops_per_token": R["lm_flops_per_token"][k], "ratio": per_token / R["lm_flops_per_token"][k]}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))

# 요약 출력
print("ceil 근사 불일치율", R["ceil_formula_mismatch"])
for k, v in R["image_tokens"].items():
    print(f"{k:34s} code:{v['code_default(max 1280토큰)']['resized']} {v['code_default(max 1280토큰)']['llm_tokens']:5d} | hub:{v['hub_default(max 16384토큰)']['resized']} {v['hub_default(max 16384토큰)']['llm_tokens']:5d}")
print(json.dumps(R["video_tokens"], ensure_ascii=False))
print(R["vla"])
for k, v in P.items():
    print(k, "vit", v["vit"]["vit_no_merger"], v["vit"]["vit_with_merger"], "lm", v["lm"]["total"], v["lm"]["non_embedding"], "total", v["total_calc"], "hub", v["hub_safetensors_total"], "diff", v["diff_vs_hub"])
for k, v in R["vit_flops"].items():
    print(k, v["patches"], f"{v['total']/1e9:.1f} GFLOPs attn_share {v['attn_share']:.2f}")
print(R["lm_flops_per_token"], R["vit_per_visual_token_vs_lm"])
print(R["table7_fixed_tokens_square"], R["rules"])
