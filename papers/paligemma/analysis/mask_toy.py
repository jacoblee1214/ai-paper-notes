"""PaliGemma의 prefix-LM 마스크 식 검산과 작은 numpy 토이 (CPU, 1초 미만, 무작위 가중치).

공식 코드 big_vision/models/proj/paligemma/paligemma.py 의 make_attn_mask 를 numpy로 옮겨 쓴다(줄 27-51):
    cumsum = cumsum(mask_ar);  attn[b,i,j] = cumsum[j] <= cumsum[i]  AND  input_mask[i] AND input_mask[j]
mask_ar는 이미지 토큰에 0(코드가 앞에 0으로 채움, 줄 123-127), 전이 설정의 접두·SEP에 0, 접미에 1, 패딩에 1(common.py 36-41)이다.

T1 같은 마스크를 논문 서술(이미지·접두 전체 양방향, 접미 인과, 패딩도 인과 영역)에서 독립적으로 만든 것과 비교한다.
T2 논문 그림 2의 토큰 배열(img×3, bos, inp×2, sep, out×2, eos, pad)의 11×11 행렬을 그린다.
T3 정보 흐름: 접미를 바꿔도 이미지·접두 표현은 그대로(KV 캐시가 성립), 접두를 바꾸면 이미지 표현이 바뀜(lookahead),
   접미 안은 인과. 인과(접두 AR, 이미지 AR) 절제 마스크에서는 어떻게 다른지.
T4 prefill(접두 블록을 한 번에) + 접미 토큰 하나씩 확장 == 전체 순전파 (공식 prefill_cache/extend_cache 설계가 성립하는 이유).
T5 손실 마스크와 한 칸 이동: 감독되는 위치와 토큰 수, 패딩 제외, 예제당 정규화.
T6 π0(openpi)의 3블록 마스크를 같은 함수로 만들어 PaliGemma와 무엇이 같고 다른지 본다.
출력: mask_results.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def make_attn_mask(input_mask, mask_ar):
    """big_vision make_attn_mask의 numpy 이식. input_mask: bool[N], mask_ar: int[N] -> bool[N,N] (행=질의, 열=키)."""
    cumsum = np.cumsum(mask_ar)
    attn = cumsum[None, :] <= cumsum[:, None]
    valid = input_mask[None, :] & input_mask[:, None]
    return attn & valid


def paper_rule(n_img, n_pre, n_sep, n_suf, n_pad, pads_valid=True):
    """논문 §3.1 서술에서 직접 만든 마스크: 앞 블록(이미지+접두+SEP)은 서로 전부, 그 뒤(접미, 패딩)는 인과."""
    n = n_img + n_pre + n_sep + n_suf + n_pad
    blk = n_img + n_pre + n_sep
    m = np.zeros((n, n), bool)
    m[:blk, :blk] = True
    for i in range(blk, n):
        m[i, : i + 1] = True
    if not pads_valid:
        v = np.ones(n, bool)
        v[n - n_pad:] = False
        m &= v[None, :] & v[:, None]
    return m


def ar_flags(n_img, n_pre, n_sep, n_suf, n_pad):
    # 이미지 0, 접두(bos 포함) 0, SEP 0, 접미(EOS 포함) 1, 패딩 1
    return np.array([0] * (n_img + n_pre + n_sep) + [1] * (n_suf + n_pad))


res = {}

# ---------------------------------------------------------------- T1
cases = []
for n_img, n_pre, n_sep, n_suf, n_pad in [(3, 3, 1, 3, 1), (4, 2, 1, 5, 0), (16, 7, 1, 9, 6), (1, 1, 1, 1, 1)]:
    ar = ar_flags(n_img, n_pre, n_sep, n_suf, n_pad)
    n = len(ar)
    got_train = make_attn_mask(np.ones(n, bool), ar)                       # 학습: input_mask 전부 True, 패딩도 인과 영역
    exp_train = paper_rule(n_img, n_pre, n_sep, n_suf, n_pad, pads_valid=True)
    v = np.ones(n, bool)
    v[n - n_pad:] = False
    got_eval = make_attn_mask(v, ar)                                        # 평가: 패딩 input_mask=0
    exp_eval = paper_rule(n_img, n_pre, n_sep, n_suf, n_pad, pads_valid=False)
    cases.append({"shape": [n_img, n_pre, n_sep, n_suf, n_pad], "train_equal": bool((got_train == exp_train).all()),
                  "eval_equal": bool((got_eval == exp_eval).all()), "allowed_pairs_train": int(got_train.sum())})
    assert (got_train == exp_train).all() and (got_eval == exp_eval).all()
res["T1_code_vs_paper_rule"] = cases

# 공식 docstring의 세 예
assert (make_attn_mask(np.ones(6, bool), np.array([1] * 6)) == np.tril(np.ones((6, 6), bool))).all()
pl = make_attn_mask(np.ones(6, bool), np.array([0, 0, 0, 1, 1, 1]))
assert pl[:3, :3].all() and not pl[:3, 3:].any() and (pl[3:, 3:] == np.tril(np.ones((3, 3), bool))).all() and pl[3:, :3].all()
blk = make_attn_mask(np.ones(10, bool), np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0]))
bid = np.cumsum([1, 0, 1, 0, 1, 0, 0, 1, 0, 0])
assert (blk == (bid[None, :] <= bid[:, None])).all()
res["T1_docstring_examples"] = "causal, prefix-lm, 4-block causal 세 예 모두 일치"

# ---------------------------------------------------------------- T2 그림 2
names = ["img1", "img2", "img3", "[bos]", "inp1", "inp2", "[sep]", "out1", "out2", "[eos]", "[pad]"]
ar2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])   # bos는 접두에 포함: img(3) bos inp1 inp2 sep | out1 out2 eos pad
m2 = make_attn_mask(np.ones(11, bool), ar2)
res["T2_fig2_matrix"] = {"tokens": names, "mask_ar": ar2.tolist(), "attend_rows_over_cols": m2.astype(int).tolist()}
print("T2 그림 2 토큰 배열의 마스크 (행이 열을 보면 1):")
print("          " + " ".join(f"{n[:5]:>5}" for n in names))
for i, nm in enumerate(names):
    print(f"{nm:>9} " + " ".join(f"{int(x):>5}" for x in m2[i]))
res["T2_note"] = ("논문 그림 2의 PDF 렌더링은 img1·img2 행과 [pad] 행 첫 열의 체크 아이콘이 빠져 보인다(poppler·MuPDF가 ExtGState 'a0'를 못 읽음). "
                  "본문 서술(§3.1 '이미지와 접두 전체에 마스크 없는 어텐션')과 위 식을 따르면 이 칸들은 모두 1이다.")

# ---------------------------------------------------------------- T3, T4 작은 트랜스포머
D, H, L = 16, 2, 2
Wq, Wk, Wv, Wo, W1, W2 = ([rng.normal(0, 0.3, (D, D)) for _ in range(L)] for _ in range(6))


def rms(x):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)


def attn(q, k, v, allow):
    """q:[Tq,D] k,v:[Tk,D] allow:[Tq,Tk] bool. 헤드 2개."""
    out = np.zeros_like(q)
    dh = D // H
    for h in range(H):
        s = q[:, h * dh:(h + 1) * dh] @ k[:, h * dh:(h + 1) * dh].T / np.sqrt(dh)
        s = np.where(allow, s, -1e30)
        s = np.exp(s - s.max(-1, keepdims=True))
        s = s * allow
        p = s / s.sum(-1, keepdims=True)
        out[:, h * dh:(h + 1) * dh] = p @ v[:, h * dh:(h + 1) * dh]
    return out


def forward(x, mask):
    """전체 순전파. 층마다 (입력, 키, 값)을 돌려 KV 캐시 시험에 쓴다."""
    caches = []
    for l in range(L):
        h = rms(x)
        k, v = h @ Wk[l], h @ Wv[l]
        caches.append((k, v))
        x = x + attn(h @ Wq[l], k, v, mask) @ Wo[l]
        x = x + np.tanh(rms(x) @ W1[l]) @ W2[l]
    return x, caches


n_img, n_pre, n_sep, n_suf = 6, 4, 1, 5      # 접미 5개(마지막은 EOS 역할)
blk = n_img + n_pre + n_sep
N = blk + n_suf
x0 = rng.normal(0, 1, (N, D))
ar = ar_flags(n_img, n_pre, n_sep, n_suf, 0)
M_plm = make_attn_mask(np.ones(N, bool), ar)
M_causal = np.tril(np.ones((N, N), bool))                                   # 이미지·접두까지 인과 (절제)
ar_pref = np.array([0] * n_img + [1] * (N - n_img))                         # 접두에도 AR (이미지만 양방향)
M_pref_ar = make_attn_mask(np.ones(N, bool), ar_pref)

y_plm, caches_plm = forward(x0, M_plm)


def changed(mask, pos, eps=1.0):
    x1 = x0.copy()
    x1[pos] += eps * rng.normal(size=D)
    y0, _ = forward(x0, mask)
    y1, _ = forward(x1, mask)
    diff = np.abs(y1 - y0).max(-1)
    return [int(i) for i in np.where(diff > 1e-12)[0]]


t3 = {}
suf_pos = blk + 2                             # 접미 셋째 토큰을 바꾼다
t3["perturb_suffix3__changed_positions__prefixLM"] = changed(M_plm, suf_pos)
t3["perturb_suffix3__changed_positions__causal"] = changed(M_causal, suf_pos)
pre_pos = n_img + 2                           # 접두 셋째 토큰을 바꾼다
t3["perturb_prefix_tok__changed_positions__prefixLM"] = changed(M_plm, pre_pos)
t3["perturb_prefix_tok__changed_positions__prefixAR"] = changed(M_pref_ar, pre_pos)
t3["perturb_prefix_tok__changed_positions__fullcausal"] = changed(M_causal, pre_pos)
t3["layout"] = {"image": [0, n_img - 1], "prefix": [n_img, n_img + n_pre - 1], "sep": n_img + n_pre, "suffix": [blk, N - 1]}
img_idx = set(range(n_img))
t3["image_tokens_see_prefix_change__prefixLM"] = img_idx <= set(t3["perturb_prefix_tok__changed_positions__prefixLM"])
t3["image_tokens_see_prefix_change__prefixAR"] = bool(img_idx & set(t3["perturb_prefix_tok__changed_positions__prefixAR"]))
t3["image_tokens_see_prefix_change__fullcausal"] = bool(img_idx & set(t3["perturb_prefix_tok__changed_positions__fullcausal"]))
# 접미를 바꿔도 접두 블록(이미지+접두+SEP) 은닉 상태는 정확히 같은가
x1 = x0.copy(); x1[blk:] = rng.normal(0, 1, (n_suf, D))
y1, _ = forward(x1, M_plm)
t3["prefix_block_hidden_identical_when_suffix_replaced"] = bool(np.abs(y1[:blk] - y_plm[:blk]).max() == 0.0)
assert t3["prefix_block_hidden_identical_when_suffix_replaced"]
assert t3["image_tokens_see_prefix_change__prefixLM"] and not t3["image_tokens_see_prefix_change__prefixAR"] and not t3["image_tokens_see_prefix_change__fullcausal"]
assert set(t3["perturb_suffix3__changed_positions__prefixLM"]) == set(range(suf_pos, N))   # 접미는 인과: 자기와 뒤만 바뀜
res["T3_information_flow"] = t3

# ---- T4 KV 캐시
# 1) prefill: 접두 블록(이미지+접두+SEP)만 한 번에 계산(블록 안 양방향) → 층별 K,V 보관
xb = x0[:blk]
mb = M_plm[:blk, :blk]
cache_k = [None] * L
cache_v = [None] * L
h_blk = xb
for l in range(L):
    h = rms(h_blk)
    cache_k[l], cache_v[l] = h @ Wk[l], h @ Wv[l]
    h_blk = h_blk + attn(h @ Wq[l], cache_k[l], cache_v[l], mb) @ Wo[l]
    h_blk = h_blk + np.tanh(rms(h_blk) @ W1[l]) @ W2[l]
# 2) extend: 접미 토큰을 하나씩. 새 토큰의 질의는 캐시된 모든 키와 자기 자신(공식 extend_cache의 마스크: 사용 중인 칸 전부)
outs = []
for t in range(blk, N):
    xt = x0[t:t + 1]
    for l in range(L):
        h = rms(xt)
        k, v = h @ Wk[l], h @ Wv[l]
        cache_k[l] = np.vstack([cache_k[l], k]); cache_v[l] = np.vstack([cache_v[l], v])
        allow = np.ones((1, cache_k[l].shape[0]), bool)
        xt = xt + attn(h @ Wq[l], cache_k[l], cache_v[l], allow) @ Wo[l]
        xt = xt + np.tanh(rms(xt) @ W1[l]) @ W2[l]
    outs.append(xt[0])
y_inc = np.array(outs)
err = float(np.abs(y_inc - y_plm[blk:]).max())
res["T4_prefill_extend_vs_full_forward"] = {"max_abs_diff_suffix_hidden": err, "layers": L, "dim": D, "ok": err < 1e-10}
assert err < 1e-10
print(f"T4 prefill+extend 대 전체 순전파: 접미 은닉 상태 최대 차이 {err:.2e}")
print("T3 접미 셋째 토큰을 바꾸면 바뀌는 위치(prefix-LM):", t3["perturb_suffix3__changed_positions__prefixLM"], "(접미 인과 → 자기와 뒤)")
print("T3 접두 토큰을 바꾸면 바뀌는 위치 prefix-LM:", t3["perturb_prefix_tok__changed_positions__prefixLM"])
print("                                  접두 AR:", t3["perturb_prefix_tok__changed_positions__prefixAR"])
print("                                  전체 인과:", t3["perturb_prefix_tok__changed_positions__fullcausal"])

# ---------------------------------------------------------------- T5 손실
text = ["bos", "p1", "p2", "sep", "s1", "s2", "eos", "pad", "pad"]          # 이미지 토큰은 text에 없다
mask_ar_t = [0, 0, 0, 0, 1, 1, 1, 1, 1]
mask_loss_t = [0, 0, 0, 0, 1, 1, 1, 0, 0]
# 학습 코드: 입력 = text[:-1], 목표 = text[1:], 마스크 = mask_loss[1:]
tgt = text[1:]
ml = mask_loss_t[1:]
sup_pos = [i for i, m in enumerate(ml) if m]           # 입력 위치 i의 출력이 tgt[i]를 예측
res["T5_loss_mask"] = {
    "targets": tgt, "mask_loss_shifted": ml, "supervised_positions_in_input": sup_pos,
    "supervised_targets": [tgt[i] for i in sup_pos],
    "predicting_first_suffix_token_from": text[:-1][sup_pos[0]],
    "n_supervised": len(sup_pos), "n_suffix_tokens_incl_eos": 3,
    "prefix_tokens_supervised": False, "pad_supervised": False,
}
assert [tgt[i] for i in sup_pos] == ["s1", "s2", "eos"] and text[:-1][sup_pos[0]] == "sep"
# 예제당 정규화: loss_i = sum(token_nll·mask)/max(sum(mask),1), 배치 평균 (train.py 줄 336-355). 토큰 평균과 다르다.
nll = [np.array([2.0] * 3), np.array([1.0] * 30)]       # 짧은 접미(3토큰, 손실 2.0)와 긴 접미(30토큰, 손실 1.0)
per_example = np.mean([x.mean() for x in nll])
per_token = np.concatenate(nll).mean()
res["T5_normalisation_example"] = {"per_example_mean_then_batch_mean": round(float(per_example), 3), "global_token_mean": round(float(per_token), 3),
                                   "note": "코드는 앞쪽(예제당 평균). 긴 출력(탐지·분할·OCR 전체 읽기)의 토큰 가중이 1/길이로 줄어든다. 논문은 정규화를 쓰지 않는다."}

# ---------------------------------------------------------------- T6 π0
# openpi pi0.py: 이미지 ar=False, 언어(BOS+프롬프트+'\n') ar=False, 상태 1개 ar=True, 행동 청크 첫 토큰 ar=True 나머지 False
n_i, n_l, H_a = 4, 3, 4
ar_pi0 = np.array([0] * n_i + [0] * n_l + [1] + [1] + [0] * (H_a - 1))
Np = len(ar_pi0)
valid = np.ones(Np, bool)
m_pi0 = make_attn_mask(valid, ar_pi0)
blocks = {"prefix(img+lang)": (0, n_i + n_l), "state": (n_i + n_l, n_i + n_l + 1), "action": (n_i + n_l + 1, Np)}
def sees(a, b):
    (a0, a1), (b0, b1) = blocks[a], blocks[b]
    return bool(m_pi0[a0:a1, b0:b1].all()), bool(m_pi0[a0:a1, b0:b1].any())
t6 = {"mask_ar": ar_pi0.tolist(), "matrix": m_pi0.astype(int).tolist(),
      "prefix_sees_state": sees("prefix(img+lang)", "state"), "prefix_sees_action": sees("prefix(img+lang)", "action"),
      "state_sees_prefix": sees("state", "prefix(img+lang)"), "state_sees_action": sees("state", "action"),
      "action_sees_prefix": sees("action", "prefix(img+lang)"), "action_sees_state": sees("action", "state"),
      "action_block_is_bidirectional": bool(m_pi0[blocks["action"][0]:, blocks["action"][0]:].all())}
assert t6["prefix_sees_state"] == (False, False) and t6["prefix_sees_action"] == (False, False)
assert t6["state_sees_prefix"][0] and t6["action_sees_prefix"][0] and t6["action_sees_state"][0] and t6["action_block_is_bidirectional"]
assert t6["state_sees_action"] == (False, False)
# PaliGemma 접미(인과 텍스트)와의 차이: 같은 함수, ar 배열만 다르다
ar_pg_like = np.array([0] * (n_i + n_l) + [1] * (1 + H_a))
t6["if_suffix_were_causal_like_paligemma_action_block_bidirectional"] = bool(
    make_attn_mask(valid, ar_pg_like)[blocks["action"][0]:, blocks["action"][0]:].all())
t6["source"] = ("openpi src/openpi/models/pi0.py make_attn_mask(줄 19-44, 'Adapted from big_vision', 식 동일), embed_prefix(줄 106-137: 이미지·언어 ar=False), "
                "embed_suffix(줄 140-186: 상태 ar=True, 행동 [True]+[False]*(H-1)). 정적 확인, 실행 안 함.")
res["T6_pi0_mask_from_same_function"] = t6
print("T6 π0 3블록:", {k: v for k, v in t6.items() if k not in ('matrix', 'source', 'mask_ar')})

(HERE / "mask_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("모든 단언 통과")
