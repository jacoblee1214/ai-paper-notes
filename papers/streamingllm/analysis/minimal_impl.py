"""StreamingLLM(arXiv:2309.17453)의 롤링 KV 캐시를 작은 PyTorch 모듈로 짠 구현.

구성
  MiniLM            Pythia(GPT-NeoX, RoPE 일부 차원, 병렬 잔차)와 GPT-2(학습된 절대 위치)를 한 코드로 돌리는 디코더.
                    HF 가중치를 그대로 옮겨 오고, HF 로짓과 같은지 테스트로 확인한다.
  attend()          한 번에(병렬로) 계산하는 스트리밍 attention. 질의 t가 보는 키는
                    "처음 s개(싱크) + 최근 w개(자기 포함)"이고, 싱크는 캐시 안 위치로 회전한다.
  RollingKVCache    토큰을 하나씩 넣는 참조 구현. 회전하기 전의 키를 캐시에 두고, 매 단계 캐시 안 위치로 다시 회전한다.
                    논문 3.2절의 설명(그림 4)을 그대로 따른다. attend()와 같은 값을 내는지 테스트로 확인한다.
  nll_recompute()   슬라이딩 윈도 + 재계산 기준선(논문 그림 1c). 윈도마다 처음부터 다시 계산한다.

캐시 크기 C = s + w는 현재 토큰을 포함해 센다(공식 코드는 현재 토큰 하나를 더 본다. 분석 04 D1).
위치 규칙: 캐시 안 위치 = 캐시 안의 순번. 질의 위치 = min(t, C-1).
  RoPE: 키를 저장할 때는 회전하지 않고, 주의를 계산할 때 순번으로 회전한다(상대 거리가 캐시 안에서 연속).
  GPT-2: 위치가 입력 임베딩에 더해져 K/V 안에 굳으므로 캐시 안에서 다시 매길 수 없다.
         pos_mode="orig"(원래 위치, 길이 1024 이하만 가능) 또는 "slot"(넣을 때의 순번 min(t, C-1))만 가능하다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _acc(x):
    """누적 정밀도: float64 입력은 float64, 그 밖은 float32."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


# ---------------------------------------------------------------- RoPE
def rope(x, pos, rot, base=10000.0):
    """x: (..., T, hd), pos: (T,). 앞의 rot 차원만 HF GPT-NeoX와 같은 rotate_half 방식으로 회전."""
    if rot == 0:
        return x
    ft = torch.float64 if x.dtype == torch.float64 else torch.float32
    inv = 1.0 / (base ** (torch.arange(0, rot, 2, device=x.device, dtype=ft) / rot))
    ang = pos.to(ft)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., : rot // 2], x[..., rot // 2: rot]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rotated, x[..., rot:]], dim=-1)


# ---------------------------------------------------------------- 모델
class Layer(nn.Module):
    def __init__(self, d, n_head, d_ff, bias_qkv=True, eps=1e-5):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d, eps=eps), nn.LayerNorm(d, eps=eps)
        self.qkv, self.dense = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc_in, self.fc_out = nn.Linear(d, d_ff), nn.Linear(d_ff, d)


class MiniLM(nn.Module):
    """kind="neox": RoPE(앞 rot 차원), 병렬 잔차, 정확한 GELU. kind="gpt2": 절대 위치, 직렬 잔차, tanh GELU."""

    def __init__(self, kind, n_layer, n_head, d, vocab, max_pos=2048, rot=0, rope_base=10000.0,
                 d_ff=None, eps=1e-5):
        super().__init__()
        assert kind in ("neox", "gpt2")
        self.kind, self.n_head, self.d, self.hd = kind, n_head, d, d // n_head
        self.rot, self.rope_base, self.max_pos = rot, rope_base, max_pos
        self.sink_pos = "cache"   # "cache": 질의를 캐시 안 위치 min(t, C-1)로 회전(논문). "orig": 원래 위치 t로 회전(소거 실험)
        self.emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_pos, d) if kind == "gpt2" else None
        self.layers = nn.ModuleList([Layer(d, n_head, d_ff or 4 * d, eps=eps) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d, eps=eps)
        self.head = nn.Linear(d, vocab, bias=False)

    # ----- HF에서 가중치 복사
    @classmethod
    def from_hf(cls, hf):
        c = hf.config
        if c.model_type == "gpt_neox":
            hd = c.hidden_size // c.num_attention_heads
            rp = getattr(c, "rope_parameters", None)   # transformers 5: rope_parameters, 이전 판: rotary_pct
            frac = rp.get("partial_rotary_factor", 1.0) if rp else c.rotary_pct
            base = rp["rope_theta"] if rp else c.rotary_emb_base
            rot = int(hd * frac)
            assert c.use_parallel_residual
            m = cls("neox", c.num_hidden_layers, c.num_attention_heads, c.hidden_size, c.vocab_size,
                    c.max_position_embeddings, rot, float(base), c.intermediate_size, c.layer_norm_eps)
            sd = hf.state_dict()
            m.emb.weight.data.copy_(sd["gpt_neox.embed_in.weight"])
            for i, L in enumerate(m.layers):
                p = f"gpt_neox.layers.{i}."
                for mine, theirs in [(L.ln1, "input_layernorm"), (L.ln2, "post_attention_layernorm"),
                                     (L.qkv, "attention.query_key_value"), (L.dense, "attention.dense"),
                                     (L.fc_in, "mlp.dense_h_to_4h"), (L.fc_out, "mlp.dense_4h_to_h")]:
                    mine.weight.data.copy_(sd[p + theirs + ".weight"])
                    mine.bias.data.copy_(sd[p + theirs + ".bias"])
            m.ln_f.weight.data.copy_(sd["gpt_neox.final_layer_norm.weight"])
            m.ln_f.bias.data.copy_(sd["gpt_neox.final_layer_norm.bias"])
            m.head.weight.data.copy_(sd["lm_head.weight" if "lm_head.weight" in sd else "embed_out.weight"])
        elif c.model_type == "gpt2":
            m = cls("gpt2", c.n_layer, c.n_head, c.n_embd, c.vocab_size, c.n_positions, 0, 0.0,
                    c.n_inner or 4 * c.n_embd, c.layer_norm_epsilon)
            sd = hf.state_dict()
            m.emb.weight.data.copy_(sd["transformer.wte.weight"])
            m.pos_emb.weight.data.copy_(sd["transformer.wpe.weight"])
            for i, L in enumerate(m.layers):
                p = f"transformer.h.{i}."
                for mine, theirs in [(L.ln1, "ln_1"), (L.ln2, "ln_2")]:
                    mine.weight.data.copy_(sd[p + theirs + ".weight"])
                    mine.bias.data.copy_(sd[p + theirs + ".bias"])
                for mine, theirs in [(L.qkv, "attn.c_attn"), (L.dense, "attn.c_proj"),
                                     (L.fc_in, "mlp.c_fc"), (L.fc_out, "mlp.c_proj")]:  # Conv1D: (in, out)
                    mine.weight.data.copy_(sd[p + theirs + ".weight"].T)
                    mine.bias.data.copy_(sd[p + theirs + ".bias"])
            m.ln_f.weight.data.copy_(sd["transformer.ln_f.weight"])
            m.ln_f.bias.data.copy_(sd["transformer.ln_f.bias"])
            m.head.weight.data.copy_(sd["transformer.wte.weight"])
        else:
            raise ValueError(c.model_type)
        return m.eval()

    # ----- 층 단위 계산
    def split_qkv(self, L, h):
        B, T, _ = h.shape
        qkv = L.qkv(h)
        if self.kind == "neox":   # HF 배치: (헤드, [q|k|v], hd)
            q, k, v = qkv.view(B, T, self.n_head, 3 * self.hd).chunk(3, dim=-1)
        else:                     # GPT-2: [q|k|v] 이어붙임
            q, k, v = (t.view(B, T, self.n_head, self.hd) for t in qkv.chunk(3, dim=-1))
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # (B,H,T,hd)

    def mlp(self, L, h):
        return L.fc_out(F.gelu(L.fc_in(h), approximate="tanh" if self.kind == "gpt2" else "none"))

    def block(self, L, x, attn):
        """attn(q, k, v) -> (B,H,T,hd). 회전 전의 q, k, v를 넘긴다."""
        q, k, v = self.split_qkv(L, L.ln1(x))
        B, _, T, _ = q.shape
        a = L.dense(attn(q, k, v).transpose(1, 2).reshape(B, T, self.d))
        if self.kind == "neox":
            return x + a + self.mlp(L, L.ln2(x))
        x = x + a
        return x + self.mlp(L, L.ln2(x))

    def embed(self, ids, pos_ids):
        x = self.emb(ids)
        return x + self.pos_emb(pos_ids) if self.kind == "gpt2" else x

    def positions(self, T, spec, pos_mode, device):
        """GPT-2 입력 위치. spec=(s, w)이면 pos_mode가 'orig'(0..T-1) 또는 'slot'(min(t, C-1))."""
        t = torch.arange(T, device=device)
        if spec is not None and pos_mode == "slot":
            t = t.clamp(max=spec[0] + spec[1] - 1)
        if self.kind == "gpt2":
            assert int(t.max()) < self.max_pos, "GPT-2는 위치 임베딩이 max_pos개뿐이라 이보다 긴 원래 위치를 쓸 수 없다"
        return t

    # ----- 병렬 attention (전체 또는 싱크+윈도)
    def attend(self, q, k, v, spec=None, pos0=0, qb=256):
        """spec=None: 전체 인과 attention. spec=(s, w): 질의 t는 키 [0,s)와 [max(s, t-w+1), t]를 본다.
        RoPE일 때 윈도 키는 원래 위치로 회전해도 상대 거리가 캐시 안 거리와 같고, 싱크 키는 위치 0..s-1,
        질의는 min(t, C-1)로 회전한다(캐시 안 위치)."""
        B, H, T, hd = q.shape
        dev, scale = q.device, hd ** -0.5
        pos = torch.arange(T, device=dev) + pos0
        qr, kr = rope(q, pos, self.rot, self.rope_base), rope(k, pos, self.rot, self.rope_base)
        out = torch.empty_like(q)
        if spec is not None:
            s, w = spec
            C = s + w
            if s > 0:
                ks = rope(k[:, :, :s], torch.arange(s, device=dev), self.rot, self.rope_base)
                qc = qr if self.sink_pos == "orig" else rope(q, torch.arange(T, device=dev).clamp(max=C - 1), self.rot, self.rope_base)
        for a in range(0, T, qb):
            b = min(a + qb, T)
            tq = torch.arange(a, b, device=dev)[:, None]
            lo = 0 if spec is None else max(spec[0], a - spec[1] + 1)
            tk = torch.arange(lo, b, device=dev)[None]
            vis = tk <= tq
            if spec is not None:
                vis = vis & (tk >= tq - spec[1] + 1)
            sc = (qr[:, :, a:b] @ kr[:, :, lo:b].transpose(-1, -2)) * scale
            sc = sc.masked_fill(~vis, float("-inf"))
            vv = v[:, :, lo:b]
            if spec is not None and spec[0] > 0:
                s = spec[0]
                ss = (qc[:, :, a:b] @ ks.transpose(-1, -2)) * scale
                ss = ss.masked_fill(~(torch.arange(s, device=dev)[None] <= tq), float("-inf"))
                sc, vv = torch.cat([ss, sc], -1), torch.cat([v[:, :, :s], vv], 2)
            out[:, :, a:b] = torch.softmax(sc, -1, dtype=_acc(sc)) @ vv
        return out

    def hidden(self, ids, spec=None, pos_mode="orig", pos0=0):
        pos_ids = self.positions(ids.shape[1], spec, pos_mode, ids.device) + (pos0 if self.kind == "gpt2" else 0)
        x = self.embed(ids, pos_ids)
        for L in self.layers:
            x = self.block(L, x, lambda q, k, v: self.attend(q, k, v, spec, pos0))
        return self.ln_f(x)

    def logits(self, ids, spec=None, pos_mode="orig", pos0=0):
        return self.head(self.hidden(ids, spec, pos_mode, pos0))

    @torch.no_grad()
    def nll(self, ids, spec=None, pos_mode="orig", bs=4, chunk=512):
        """ids: (N, T). 반환 (N, T-1): 위치 t의 값 = 토큰 t+1을 예측하는 음의 로그우도."""
        res = []
        for i in range(0, ids.shape[0], bs):
            x = ids[i:i + bs]
            h = self.hidden(x, spec, pos_mode)
            parts = []
            for a in range(0, x.shape[1] - 1, chunk):
                b = min(a + chunk, x.shape[1] - 1)
                lg = self.head(h[:, a:b]).to(_acc(h))
                parts.append(F.cross_entropy(lg.transpose(1, 2), x[:, a + 1:b + 1], reduction="none"))
            res.append(torch.cat(parts, 1))
        return torch.cat(res, 0)


# ---------------------------------------------------------------- 슬라이딩 윈도 + 재계산
@torch.no_grad()
def nll_recompute(model, ids, C, positions, bs=64):
    """positions의 각 t에 대해 토큰 [t-C+1, t]만 처음부터 다시 계산(위치 0..C-1)하고 토큰 t+1의 nll을 낸다.
    반환 (N, len(positions)). t < C-1이면 앞에서부터 있는 만큼(=전체)을 쓴다."""
    N = ids.shape[0]
    out = torch.zeros(N, len(positions), device=ids.device, dtype=_acc(next(model.parameters())))
    for j, t in enumerate(positions):
        t = int(t)
        lo = max(0, t - C + 1)
        for i in range(0, N, bs):
            x = ids[i:i + bs, lo:t + 1]
            lg = model.logits(x)[:, -1]
            lg = lg.to(_acc(lg))
            out[i:i + bs, j] = F.cross_entropy(lg, ids[i:i + bs, t + 1], reduction="none")
    return out


# ---------------------------------------------------------------- 토큰 하나씩 롤링 KV 캐시 (참조 구현)
class RollingKVCache:
    """층마다 회전 전의 K와 V를 쌓고, 크기가 C = n_sink + window를 넘으면 싱크 뒤의 가장 오래된 항목을 뺀다."""

    def __init__(self, model, n_sink, window, pos_mode="orig"):
        self.m, self.s, self.w, self.pos_mode = model, n_sink, window, pos_mode
        self.C = n_sink + window
        self.k = [None] * len(model.layers)
        self.v = [None] * len(model.layers)
        self.kept = []    # 캐시에 남은 원래 토큰 번호(0층 기준, 테스트용)
        self.t = 0

    @torch.no_grad()
    def step(self, tok):
        """tok: (B,) 다음 입력 토큰. 반환 (B, vocab) 로짓."""
        m = self.m
        pos = min(self.t, self.C - 1) if self.pos_mode == "slot" else self.t
        x = m.embed(tok[:, None], torch.tensor([pos], device=tok.device))
        self.kept.append(self.t)
        layer_idx = iter(range(len(m.layers)))

        def attn(q, k, v):
            i = next(layer_idx)
            self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k], 2)
            self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v], 2)
            n = self.k[i].shape[2]
            if n > self.C:   # 현재 토큰을 넣은 뒤 넘치면 바로 뺀다(층마다 같은 규칙)
                drop = n - self.C
                self.k[i] = torch.cat([self.k[i][:, :, :self.s], self.k[i][:, :, self.s + drop:]], 2)
                self.v[i] = torch.cat([self.v[i][:, :, :self.s], self.v[i][:, :, self.s + drop:]], 2)
                n = self.C
                if i == 0:
                    self.kept = self.kept[:self.s] + self.kept[self.s + drop:]
            cpos = torch.arange(n, device=q.device)           # 캐시 안 위치
            kr = rope(self.k[i], cpos, m.rot, m.rope_base)
            qr = rope(q, cpos[-1:], m.rot, m.rope_base)
            p = torch.softmax((qr @ kr.transpose(-1, -2)) * m.hd ** -0.5, -1, dtype=_acc(q))
            return p @ self.v[i]

        for L in m.layers:
            x = m.block(L, x, attn)
        self.t += 1
        return m.head(m.ln_f(x))[:, 0]
