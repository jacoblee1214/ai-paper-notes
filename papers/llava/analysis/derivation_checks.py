"""LLaVA 식·산술 검산 (학습 없음, 수 초). 결과는 derivation_results.json.

실물 규모(CLIP ViT-L/14, Vicuna 7B/13B)의 시각 토큰 수, 투영기 파라미터, 비전 타워 파라미터, 문맥 예산, 전방 계산량,
KV 캐시, 학습 계산량의 일관성(논문 부록 C의 학습 시간과 맞는지)을 산술로 확인한다. 가정은 항목마다 적는다.
실행: python llava/analysis/derivation_checks.py  (저장소 루트에서)
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as M  # noqa: E402

OUT = Path(__file__).with_name("derivation_results.json")


def main():
    r = {}

    # 1. 시각 토큰 수 (CLS 제외)
    r["visual_tokens"] = {
        "224/14": M.num_visual_tokens(224, 14),
        "336/14": M.num_visual_tokens(336, 14),
        "with_cls_224": M.num_visual_tokens(224, 14, include_cls=True),
        "anyres_llava_next_max_tiles_5x576": 5 * M.num_visual_tokens(336, 14),
        "toy_32/4": M.num_visual_tokens(32, 4),
    }

    # 2. 투영기 파라미터
    d_vis = 1024
    proj = {}
    for name, d_lm in [("vicuna7b_4096", 4096), ("vicuna13b_5120", 5120)]:
        proj[name] = {
            "linear": M.projector_params("linear", d_vis, d_lm),
            "mlp2x_gelu": M.projector_params("mlp2x_gelu", d_vis, d_lm),
        }
    r["projector_params"] = proj
    lm = {"vicuna7b": M.llama_params(4096, 32, 11008), "vicuna13b": M.llama_params(5120, 40, 13824)}
    r["llm_params"] = lm
    r["projector_fraction_of_llm"] = {
        "7b_linear": proj["vicuna7b_4096"]["linear"] / lm["vicuna7b"],
        "7b_mlp2x": proj["vicuna7b_4096"]["mlp2x_gelu"] / lm["vicuna7b"],
        "13b_linear": proj["vicuna13b_5120"]["linear"] / lm["vicuna13b"],
        "13b_mlp2x": proj["vicuna13b_5120"]["mlp2x_gelu"] / lm["vicuna13b"],
    }

    # 3. 비전 타워 파라미터: 식 대 실제 모듈(meta 장치라 메모리를 쓰지 않는다)
    vit_formula = {
        "L/14@224": M.clip_vit_params(1024, 24, 4096, 14, 224),
        "L/14@336": M.clip_vit_params(1024, 24, 4096, 14, 336),
    }
    with torch.device("meta"):
        big = M.TinyViT(img=224, patch=14, dim=1024, depth=24, heads=16, mlp=4096)
        big336 = M.TinyViT(img=336, patch=14, dim=1024, depth=24, heads=16, mlp=4096)
    vit_module = {"L/14@224": big.tower_params(), "L/14@336": big336.tower_params()}
    r["vision_tower_params"] = {"formula": vit_formula, "module_on_meta_device": vit_module}
    r["vision_tower_fraction_of_7b"] = vit_formula["L/14@224"] / lm["vicuna7b"]

    # 4. 문맥 예산 (model_max_length=2048, 공식 학습 스크립트)
    r["context_budget_2048"] = {
        "576_visual": M.context_budget(2048, 576),
        "256_visual": M.context_budget(2048, 256),
        "2880_visual_anyres": M.context_budget(2048, 2880),   # 음수: 2048 로는 못 담는다
        "2880_visual_in_4096": M.context_budget(4096, 2880),
    }

    # 5. 전방 계산량 (FLOPs = 2 x 파라미터 x 토큰, 어텐션 제곱항은 따로)
    def lm_prefill(n_params, tokens):
        return 2 * n_params * tokens

    def vit_forward(width, layers, mlp, tokens, params):
        attn_quadratic = 4 * layers * tokens * tokens * width      # QK^T 와 AV, 곱셈 2번 x 2
        return 2 * params * tokens + attn_quadratic

    t336 = 576 + 1
    vit_flops = vit_forward(1024, 24, 4096, t336, vit_formula["L/14@336"])
    lm_flops_576 = lm_prefill(lm["vicuna7b"], 576)
    lm_flops_32 = lm_prefill(lm["vicuna7b"], 32)
    r["forward_flops_per_image"] = {
        "vit_l14_336": vit_flops,
        "lm7b_prefill_576_visual_tokens": lm_flops_576,
        "lm7b_prefill_32_query_tokens": lm_flops_32,
        "ratio_lm576_over_vit": lm_flops_576 / vit_flops,
        "ratio_lm576_over_lm32": 576 / 32,
        "note": "텍스트 토큰, 어텐션 제곱항(LM), Q-Former 자체 비용은 제외. 이미지 한 장의 시각 토큰 prefill 만",
    }

    # 6. KV 캐시 (fp16, K와 V)
    def kv_bytes(tokens, layers, hidden):
        return tokens * 2 * layers * hidden * 2

    r["kv_cache_mb_per_image_fp16"] = {
        "7b_576": kv_bytes(576, 32, 4096) / 2**20,
        "13b_576": kv_bytes(576, 40, 5120) / 2**20,
        "7b_2880": kv_bytes(2880, 32, 4096) / 2**20,
        "7b_32": kv_bytes(32, 32, 4096) / 2**20,
    }

    # 7. 학습 계산량 일관성 (부록 C: 8xA100, 1단계 4시간, 2단계 10시간). 가정이 많아 '일관성 점검'일 뿐 검증이 아니다.
    a100_bf16_peak = 312e12
    gpus = 8
    n13 = lm["vicuna13b"]
    stage1_samples = 595_000
    cap_tokens_assumed = 20            # CC3M 캡션 평균 토큰 수(가정). 논문에 없음
    s1_tokens = stage1_samples * (256 + cap_tokens_assumed)
    # 1단계: LM 동결이라 가중치 기울기는 없고 활성 기울기만 필요: 순전파 2N + 역전파 2N = 4N/토큰.
    s1_flops = 4 * n13 * s1_tokens
    s1_mfu = s1_flops / (a100_bf16_peak * gpus * 4 * 3600)
    s2_samples = 158_000 * 3
    # 2단계: 전체 미세조정은 6N/토큰, 기울기 체크포인팅이 켜져 있어 순전파를 한 번 더(8N).
    # 비전 인코더(동결)와 투영기는 무시. 평균 토큰 수는 모른다: MFU 를 가정하고 역산한다.
    s2_secs = 10 * 3600
    implied = {}
    for mfu in (0.25, 0.35, 0.45):
        tokens = mfu * a100_bf16_peak * gpus * s2_secs / (8 * n13)
        implied[f"mfu_{mfu}"] = {"total_tokens": tokens, "tokens_per_sample": tokens / s2_samples}
    r["training_compute_consistency"] = {
        "assumptions": "13B 모델, 기울기 체크포인팅(8N), bf16 피크 312 TFLOPs x 8, 캡션 20토큰 가정",
        "stage1_tokens": s1_tokens,
        "stage1_flops_4N": s1_flops,
        "stage1_implied_mfu_vs_4h": s1_mfu,
        "stage1_implied_mfu_if_6N_with_ckpt": 6 * n13 * s1_tokens / (a100_bf16_peak * gpus * 4 * 3600),
        "stage2_samples_158k_x3": s2_samples,
        "stage2_implied_tokens_per_sample_by_mfu": implied,
    }

    # 8. 토이 크기
    vis, lmt = M.TinyViT(), M.TinyLM()
    r["toy"] = {
        "visual_tokens": M.num_visual_tokens(M.IMG, M.PATCH),
        "vision_tower_params": vis.tower_params(),
        "lm_params": sum(p.numel() for p in lmt.parameters()),
        "projector_linear": M.projector_params("linear", vis.dim, lmt.dim),
        "projector_mlp2x": M.projector_params("mlp2x_gelu", vis.dim, lmt.dim),
        "vocab": M.V,
    }

    OUT.write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
