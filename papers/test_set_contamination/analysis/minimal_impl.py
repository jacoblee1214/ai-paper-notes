"""Oren et al. (2023) 교환성 기반 오염 검정의 최소 구현과 검정을 시험할 토이 언어 모델.

(1) 검정 함수: "순서 있는 문항 목록 -> 로그우도" 함수 loglik(items)만 받는다.
    - permutation_test : 전체 순열 검정 (논문 §3.1, Phipson-Smyth 보정 p값)
    - sharded_test     : 분할 순위 비교 검정 (논문 §3.2, Algorithm 1: 샤드별 정준-순열평균 차이 + 단측 t 검정)
    - permutation_z_test: 전체 순열의 로그우도에 정규분포를 맞추는 검정 (대조용. 03 §2에서 샤드 검정의 이점이
                          '샤드'가 아니라 '정규 꼬리 외삽'에서 옴을 보이려고 추가한 것. 논문에는 없음)
(2) 토이 언어 모델: numpy n-gram (Witten-Bell 보간). 문항 사이 구분자 SEP를 넣고 문맥이 SEP를 넘어 앞 문항의
    끝 토큰까지 닿게 해서(order >= 3) 문항 '순서'를 기억할 수 있게 한다. retain(보존율)은 고차 n-gram 출현을
    확률적으로만 세어 "여러 번 봐야 외운다"는 대형 LM의 성질을 흉내 낸다 (토이 가정이며 논문 값이 아님).

numpy, scipy만 쓴다. CPU 몇 초.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

SEP = 0  # 문항 구분자 토큰 (공식 코드에서는 문항 끝의 개행)


# ----------------------------------------------------------------------------
# 검정
# ----------------------------------------------------------------------------
def shard_sizes(n: int, r: int) -> list[int]:
    """공식 코드/Algorithm 1의 분할 규칙: 각 샤드에 n//r개, 앞의 n % r개 샤드에 1개씩 더."""
    return [n // r + (1 if i < n % r else 0) for i in range(r)]


def split_shards(items: list, r: int) -> list[list]:
    out, s = [], 0
    for size in shard_sizes(len(items), r):
        out.append(items[s : s + size])
        s += size
    return out


def _shuffled(items: list, rng: np.random.Generator) -> list:
    return [items[i] for i in rng.permutation(len(items))]


def permutation_test(loglik, items, m: int, rng) -> dict:
    """전체 문항 순열 검정. p = (1 + #{pi: l(pi) >= l(canonical)}) / (m + 1).

    동점은 '정준이 낮다' 쪽(>=)으로 세어 보수적으로 만든다. 논문 식은 엄격한 부등호(<)를 쓰는데, 동점이 있으면
    그쪽은 p를 줄이는 방향이라 유효성이 깨질 수 있다 (연속 로그우도에서는 차이 없음).
    """
    l_can = loglik(items)
    l_perm = np.array([loglik(_shuffled(items, rng)) for _ in range(m)])
    p = (1 + int(np.sum(l_perm >= l_can))) / (m + 1)
    return {"p": p, "l_can": l_can, "l_perm": l_perm}


def permutation_z_test(loglik, items, m: int, rng) -> dict:
    """대조용: m개 전체 순열의 로그우도에 정규분포를 맞추고 정준의 z(t) 값으로 단측 p를 낸다."""
    l_can = loglik(items)
    l_perm = np.array([loglik(_shuffled(items, rng)) for _ in range(m)])
    t = (l_can - l_perm.mean()) / l_perm.std(ddof=1)
    return {"p": float(stats.t.sf(t, df=m - 1)), "t": float(t), "l_can": l_can}


def sharded_test(loglik, items, num_shards: int, m: int, rng, ddof: int = 1) -> dict:
    """분할 순위 비교 검정 (Algorithm 1). 계산량은 문항 로그우도 평가 n*(1+m)번.

    s_i = l(정준 샤드 i) - mean_j l(순열 j의 샤드 i), s의 평균에 단측 t 검정(df = r-1).
    ddof=1이 표준 t 통계량. 공식 코드는 np.std 기본값(ddof=0)을 써서 통계량이 sqrt(r/(r-1))배 커진다.
    """
    shards = split_shards(items, num_shards)
    s = np.empty(num_shards)
    for i, shard in enumerate(shards):
        l_can = loglik(shard)
        l_shuf = np.mean([loglik(_shuffled(shard, rng)) for _ in range(m)])
        s[i] = l_can - l_shuf
    t = s.mean() / s.std(ddof=ddof) * np.sqrt(num_shards)
    return {"p": float(stats.t.sf(t, df=num_shards - 1)), "t": float(t), "s": s}


# ----------------------------------------------------------------------------
# 토이 언어 모델: Witten-Bell 보간 n-gram (numpy)
# ----------------------------------------------------------------------------
class NgramLM:
    """order차 n-gram. 토큰 id는 0..V-1 (0은 SEP). 문서 시작은 SEP로 패딩해 문맥을 채운다.

    retain: order >= 3의 n-gram 출현을 이 확률로만 센다(예측 위치마다 독립 베르누이). 1이면 일반 n-gram.
    """

    def __init__(self, vocab: int, order: int = 4, retain: float = 1.0, seed: int = 0):
        assert order >= 2
        self.V, self.N, self.retain = vocab, order, retain
        self.rng = np.random.default_rng(seed)
        self.tables: list = []

    def _ids(self, tok: np.ndarray, n: int, start: int) -> np.ndarray:
        """tok[i-n+1..i]를 정수 하나로 묶은 id (i = start..len-1)."""
        out = np.zeros(len(tok) - start, dtype=np.int64)
        for j in range(n):
            out = out * self.V + tok[start - (n - 1) + j : len(tok) - (n - 1) + j]
        return out

    def fit(self, stream: np.ndarray) -> "NgramLM":
        tok = np.concatenate([np.zeros(self.N - 1, dtype=np.int64), stream.astype(np.int64)])
        s = self.N - 1
        L = len(tok) - s
        keep = self.rng.random(L) < self.retain
        self.tables = []
        for n in range(1, self.N + 1):
            full = self._ids(tok, n, s)
            if n >= 3:
                full = full[keep]
            uk, uc = np.unique(full, return_counts=True)  # (문맥,다음토큰) 개수
            ctx = uk // self.V
            ck, inv = np.unique(ctx, return_inverse=True)
            c_ctx = np.bincount(inv, weights=uc).astype(np.int64)  # c(h)
            t_ctx = np.bincount(inv).astype(np.int64)  # T(h): 서로 다른 다음 토큰 수
            self.tables.append((uk, uc, ck, c_ctx, t_ctx))
        return self

    @staticmethod
    def _lookup(keys, vals, q):
        idx = np.searchsorted(keys, q)
        idx[idx == len(keys)] = len(keys) - 1
        hit = keys[idx] == q
        return np.where(hit, vals[idx], 0)

    def logprob_tokens(self, seq: np.ndarray) -> np.ndarray:
        """seq의 각 토큰의 로그확률. 시퀀스 앞은 SEP로 패딩한다(문서 시작)."""
        tok = np.concatenate([np.zeros(self.N - 1, dtype=np.int64), np.asarray(seq, dtype=np.int64)])
        s = self.N - 1
        uk, uc, _, _, _ = self.tables[0]
        total = uc.sum()
        c1 = self._lookup(uk, uc, tok[s:])
        p = (c1 + 1.0) / (total + self.V)  # 1-gram: add-one
        for n in range(2, self.N + 1):
            uk, uc, ck, c_ctx, t_ctx = self.tables[n - 1]
            full = self._ids(tok, n, s)
            ctx = full // self.V
            cw = self._lookup(uk, uc, full)
            ch = self._lookup(ck, c_ctx, ctx)
            th = self._lookup(ck, t_ctx, ctx)
            p = np.where(ch > 0, (cw + th * p) / np.maximum(ch + th, 1), p)
        return np.log(p)

    def loglik_items(self, items: list[np.ndarray]) -> float:
        """문항(토큰 배열) 목록을 SEP로 끝내며 이어 붙인 시퀀스의 로그우도. 검정의 loglik 함수로 쓴다."""
        parts = []
        for it in items:
            parts.append(it)
            parts.append(np.zeros(1, dtype=np.int64))
        return float(self.logprob_tokens(np.concatenate(parts)).sum())


# ----------------------------------------------------------------------------
# 합성 말뭉치와 합성 벤치마크
# ----------------------------------------------------------------------------
def zipf_probs(V: int, a: float = 1.0) -> np.ndarray:
    w = 1.0 / np.arange(1, V) ** a  # 토큰 1..V-1 (0은 SEP 전용)
    return w / w.sum()


def make_benchmark(rng, n_items: int, V: int, lo: int = 6, hi: int = 12) -> list[np.ndarray]:
    """i.i.d. 문항 n개 (각 문항은 길이 lo..hi의 토큰열). 문항 순서는 교환 가능."""
    p = zipf_probs(V)
    return [rng.choice(np.arange(1, V), size=rng.integers(lo, hi + 1), p=p) for _ in range(n_items)]


def make_background(rng, n_tokens: int, V: int, doc_len: int = 10) -> list[np.ndarray]:
    p = zipf_probs(V)
    n_docs = n_tokens // doc_len
    return [rng.choice(np.arange(1, V), size=doc_len, p=p) for _ in range(n_docs)]


def build_stream(rng, background: list, bench: list, copies: int, canonical: bool = True) -> np.ndarray:
    """배경 문서 사이 무작위 위치에 벤치마크 블록을 copies번 끼워 넣어 SEP로 이은 학습 스트림을 만든다.
    canonical=True면 매번 같은 정준 순서, False면 복사본마다 새로 섞은 순서(공개 순서가 학습에 없는 경우)."""
    blocks = []
    for _ in range(copies):
        order = list(range(len(bench))) if canonical else list(rng.permutation(len(bench)))
        blocks.append([bench[i] for i in order])
    pos = sorted(rng.integers(0, len(background) + 1, size=copies)) if copies else []
    docs, last = [], 0
    for b, p in zip(blocks, pos):
        docs += background[last:p] + b
        last = p
    docs += background[last:]
    parts = []
    for d in docs:
        parts.append(d)
        parts.append(np.zeros(1, dtype=np.int64))
    return np.concatenate(parts)
