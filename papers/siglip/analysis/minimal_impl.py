"""SigLIP(arXiv:2303.15343, v4)의 손실을 작게 구현한 토이. 논문 재현이 아니라 이해용이다.

구성 (논문 절·그림과의 대응)
  - sigmoid_loss            : 알고리즘 1, 3.2절 식. L = -(1/|B|) sum_i sum_j log sigmoid(z_ij (t x_i.y_j + b)), z_ij = +1(i=j) / -1
  - sigmoid_loss_chunked    : 3.3절, 그림 1. 장치 D개로 나눈 청크 계산(이미지 청크 고정, 텍스트 청크를 돌리며 부분 손실 합산).
                              시뮬레이션이라 한 프로세스에서 장치를 반복문으로 흉내 낸다(통신 없음).
  - softmax_loss            : 3.1절 식(= CLIP의 대칭 InfoNCE). 비교용.
  - sigmoid_grads_manual    : 손실의 기울기를 손으로 유도한 식 (autograd 대조용)
  - 해석값 함수들           : 초기 손실·편향 균형점·최적 로짓 분리·메모리 산술 (03_analysis.md의 유도와 같은 식)

모든 임베딩은 L2 정규화된 [n, d]이고 t = exp(t')는 이미 지수를 취한 값, b는 스칼라(0차원 텐서 또는 float)이다.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

T_INIT_PRIME = math.log(10.0)  # 논문 3.2절: t' 초기값 log 10 (t = 10). v1·v2는 "10"으로만 적어 t와 t'가 헷갈렸고 v3가 고침
B_INIT = -10.0                 # 논문 3.2절, 4.9절 표 4: b 초기값 -10


# ---------------------------------------------------------------- 손실

def pair_logits(zimg, ztxt, t, b):
    """로짓 행렬 [n, m] = t * zimg @ ztxt.T + b."""
    return t * zimg @ ztxt.T + b


def sigmoid_loss(zimg, ztxt, t, b):
    """알고리즘 1. labels = 2*eye(n) - ones(n), l = -sum(log_sigmoid(labels * logits)) / n."""
    n = zimg.shape[0]
    logits = pair_logits(zimg, ztxt, t, b)
    labels = 2 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1
    return -F.logsigmoid(labels * logits).sum() / n


def softmax_loss(zimg, ztxt, t):
    """3.1절: 이미지->텍스트 softmax와 텍스트->이미지 softmax의 평균 (1/(2|B|) 곱 두 항)."""
    n = zimg.shape[0]
    logits = t * zimg @ ztxt.T
    labels = torch.arange(n, device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def _block_loss(a, c, t, b, positive_block: bool):
    """이미지 청크 a [p, d]와 텍스트 청크 c [q, d]의 쌍 손실 합(정규화 전). 양성은 positive_block일 때만 대각선에 있다."""
    logits = t * a @ c.T + b
    z = -torch.ones_like(logits)
    if positive_block:
        z = z + 2 * torch.eye(a.shape[0], c.shape[0], device=z.device, dtype=z.dtype)
    return F.softplus(-z * logits).sum()  # -log sigmoid(z * logits) = softplus(-z * logits)


def sigmoid_loss_chunked(zimg, ztxt, t, b, num_devices: int, use_checkpoint: bool = False, stats: dict | None = None):
    """그림 1·3.3절의 청크 계산. 장치 d는 이미지 청크 I_d를 가지고 있고, 텍스트 청크를 한 칸씩 돌려 받는다.
      step 0: 자기 텍스트 청크 T_d (양성 b개 + 음성 b(b-1)개)
      step s: 이웃에서 온 T_{(d+s) mod D} (음성 b x b개)
    부분 손실을 모두 더해 |B|로 나눈다. 어느 시점에도 만드는 로짓 블록은 b x b 하나다(stats['max_block']에 기록).
    use_checkpoint=True이면 블록마다 역전파 때 다시 계산해 블록 로짓을 저장하지 않는다."""
    n = zimg.shape[0]
    assert n % num_devices == 0, "배치가 장치 수로 나누어떨어져야 한다"
    img = zimg.chunk(num_devices)
    txt = ztxt.chunk(num_devices)
    t = torch.as_tensor(t, dtype=zimg.dtype)
    b = torch.as_tensor(b, dtype=zimg.dtype)
    total = zimg.new_zeros(())
    biggest = 0
    for d in range(num_devices):          # 합 A: 장치마다
        for s in range(num_devices):      # 합 B: 텍스트 청크를 돌린다
            c = txt[(d + s) % num_devices]
            biggest = max(biggest, img[d].shape[0] * c.shape[0])
            if use_checkpoint:
                part = checkpoint(_block_loss, img[d], c, t, b, s == 0, use_reentrant=False)
            else:
                part = _block_loss(img[d], c, t, b, s == 0)  # 합 C: 장치 안
            total = total + part
    if stats is not None:
        stats["max_block"] = biggest
    return total / n


def count_saved_block_elements(fn, block_shape: tuple[int, int]) -> int:
    """fn()을 호출하는 동안 역전파용으로 저장된 텐서 중 block_shape(b, b)인 것의 원소 수 합.
    '블록 로짓이 역전파까지 살아 있나'를 재는 간이 지표다 (실제 메모리 측정은 아님)."""
    total = 0

    def pack(x):
        nonlocal total
        if tuple(x.shape) == tuple(block_shape):
            total += x.numel()
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        fn()
    return total


# ---------------------------------------------------------------- 손으로 유도한 기울기

def sigmoid_grads_manual(zimg, ztxt, t, b):
    """L = (1/n) sum_ij softplus(-z_ij l_ij), l_ij = t x_i.y_j + b 의 기울기 (정규화된 임베딩을 독립 변수로 본다).
      dL/dl_ij = -(1/n) z_ij sigmoid(-z_ij l_ij) =: g_ij
      dL/dx_i  = t sum_j g_ij y_j,   dL/dy_j = t sum_i g_ij x_i
      dL/dt'   = t sum_ij g_ij (x_i.y_j),   dL/db = sum_ij g_ij
    반환 (gx, gy, gt_prime, gb)."""
    n = zimg.shape[0]
    c = zimg @ ztxt.T
    logits = t * c + b
    z = 2 * torch.eye(n, dtype=c.dtype) - 1
    g = -z * torch.sigmoid(-z * logits) / n
    return t * g @ ztxt, t * g.T @ zimg, t * (g * c).sum(), g.sum()


# ---------------------------------------------------------------- 해석값 (03의 유도)

def softplus(x: float) -> float:
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def loss_const_cos(B: int, t: float, b: float, c_pos: float = 1.0, c_neg: float = 0.0) -> float:
    """모든 양성의 코사인이 c_pos, 모든 음성이 c_neg일 때의 손실 (행마다 같으므로 행 하나의 값).
    L = softplus(-(t c_pos + b)) + (B-1) softplus(t c_neg + b)."""
    return softplus(-(t * c_pos + b)) + (B - 1) * softplus(t * c_neg + b)


def bias_grad_const_cos(B: int, t: float, b: float, c_pos: float = 1.0, c_neg: float = 0.0) -> float:
    """dL/db = -sigmoid(-(t c_pos + b)) + (B-1) sigmoid(t c_neg + b) (행 하나 기준)."""
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    return -sig(-(t * c_pos + b)) + (B - 1) * sig(t * c_neg + b)


def bias_balance_point(B: int) -> float:
    """c = 0 근처(로짓이 b뿐)에서 dL/db = 0 이 되는 b: sigmoid(b)(B-1) = sigmoid(-b)  =>  b* = -ln(B-1)."""
    return -math.log(B - 1)


def optimal_orthogonal(B: int, t: float, iters: int = 200) -> dict:
    """c_pos = 1, c_neg = 0 (임베딩이 서로 직교)일 때 b에 대한 최적. dL/db가 b에서 단조 증가하므로 이분법으로 푼다.
    큰 t의 근사: b* ~ -t/2 - ln(B-1)/2, 양성 로짓 ~ t/2 - ln(B-1)/2, 음성 로짓 ~ -t/2 - ln(B-1)/2, 최소 손실 ~ 2 sqrt(B-1) e^{-t/2}."""
    lo, hi = -10 * t - 100.0, 10.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if bias_grad_const_cos(B, t, mid) > 0:
            hi = mid
        else:
            lo = mid
    b = (lo + hi) / 2
    return {"b_star": b, "logit_pos": t + b, "logit_neg": b, "loss": loss_const_cos(B, t, b),
            "approx_b": -t / 2 - math.log(B - 1) / 2, "approx_loss": 2 * math.sqrt(B - 1) * math.exp(-t / 2)}


def softmax_loss_const_cos(B: int, t: float, c_pos: float = 1.0, c_neg: float = 0.0) -> float:
    """한쪽 방향 softmax 손실: log(1 + (B-1) exp(-t (c_pos - c_neg)))."""
    return math.log1p((B - 1) * math.exp(-t * (c_pos - c_neg)))


def memory_elements(B: int, D: int) -> dict:
    """한 장치가 한 시점에 만드는 로짓 원소 수와 임베딩 통신량(원소 수, 임베딩 차원 d를 곱하기 전).
      naive      : |B| x |B| (모든 임베딩을 모은 뒤 전체 행렬)
      clip_local : 이미지 b개 x 전체 텍스트 |B|개 (장치별 로짓 + 두 방향 softmax는 모은 임베딩이 필요)
      sigmoid    : b x b (알고리즘 1의 청크 하나, 그림 1)
    통신(장치가 받는 임베딩 벡터 수): 두 방향 softmax는 이미지·텍스트 all-gather 각 (D-1)b, 청크 sigmoid는 텍스트 청크 (D-1)번 순환 (D-1)b."""
    b = B // D
    return {"b": b, "naive": B * B, "clip_local": b * B, "sigmoid_chunk": b * b,
            "recv_vectors_softmax_two_gathers": 2 * (D - 1) * b, "recv_vectors_sigmoid_permutes": (D - 1) * b}
