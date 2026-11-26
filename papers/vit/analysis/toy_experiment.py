"""토이: 귀납 편향이 적은 ViT 는 데이터가 적으면 작은 CNN 보다 불리한가 (논문 §4.2-4.3 의 방향만 확인).

직접 만든 합성 이미지(32x32, 6가지 도형, 위치·크기·밝기·잡음이 무작위)로 학습 세트 크기 n 을 바꾼다.
네 팔: {작은 CNN, 작은 ViT} x {증강 없음, 이동 증강(±4픽셀)}. 같은 optimizer, 같은 스텝 수, 같은 weight decay, 시드 5개.
이 실험은 논문의 JFT 규모 실험이 아니고 모델이 아주 작아서, 논문을 재현하지 않는다(방향 확인용).

실행: python analysis/toy_experiment.py [--quick]  (논문 폴더 vit/ 에서)  -> toy_results.json
(MPS 가 있으면 MPS, 없으면 CPU. 이 맥에서 약 15분. 시험 세트와 학습 세트는 numpy 시드로 고정, 학습은 torch 시드로 고정하지만 MPS 연산의 비결정성이 조금 있다.)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as M  # noqa: E402

HERE = Path(__file__).resolve().parent
SIZE = 32
CLASSES = ["square", "disc", "plus", "frame", "cross_x", "ring"]
NOISE = 0.35
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def make_images(n, rng):
    """위치(중심), 크기(반변 4-7), 밝기(0.6-1.0), 잡음 N(0, NOISE^2). 도형은 항상 이미지 안에 들어간다."""
    labels = rng.integers(0, len(CLASSES), n)
    half = rng.integers(4, 8, n)  # 반변
    cy = np.array([rng.integers(h, SIZE - h) for h in half])
    cx = np.array([rng.integers(h, SIZE - h) for h in half])
    amp = rng.uniform(0.6, 1.0, n)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dy = yy[None] - cy[:, None, None]
    dx = xx[None] - cx[:, None, None]
    h = half[:, None, None]
    ady, adx = np.abs(dy), np.abs(dx)
    box = (ady <= h) & (adx <= h)
    r = np.sqrt(dy ** 2 + dx ** 2)
    thick = np.maximum(1, h // 3)
    masks = np.stack([
        box,                                                    # square
        r <= h + 0.5,                                           # disc
        ((ady <= thick) & (adx <= h)) | ((adx <= thick) & (ady <= h)),  # plus
        box & ~((ady <= h - thick) & (adx <= h - thick)),       # frame
        (np.abs(ady - adx) <= thick) & box,                # cross_x
        (r <= h + 0.5) & (r > h - thick - 0.5),                 # ring
    ])  # (6, n, H, W)
    m = masks[labels, np.arange(n)]
    img = m * amp[:, None, None] + rng.normal(0, NOISE, (n, SIZE, SIZE))
    return torch.tensor(img[:, None], dtype=torch.float32), torch.tensor(labels)


class SmallCNN(nn.Module):
    """3x3 conv 4층(채널 32-64-64-96, stride 2 두 번) + GAP + 선형. 약 0.11M 파라미터."""

    def __init__(self, nc=len(CLASSES)):
        super().__init__()
        ch = [1, 32, 64, 64, 96]
        st = [1, 2, 1, 2]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, st[i], 1), nn.GroupNorm(8, ch[i + 1]), nn.GELU()]
        self.f = nn.Sequential(*layers)
        self.head = nn.Linear(96, nc)

    def forward(self, x):
        return self.head(self.f(x).mean((2, 3)))


def small_vit():
    # P=4 -> 64 패치 + [CLS], D=64, 4층, 4헤드, MLP 128. 표 1의 B 를 줄인 모양(평탄한 구조는 그대로)
    return M.ViT(img_size=SIZE, patch=4, in_ch=1, num_classes=len(CLASSES), depth=4, dim=64, mlp=128, heads=4)


def shift_aug(x, rng_t, max_shift=4):
    """이미지별 무작위 정수 이동(0 채움). 위치 불변성을 증강으로 알려 주는 쪽. x: (B, 1, H, W)"""
    b = x.shape[0]
    pad = F.pad(x, (max_shift,) * 4)[:, 0]
    dy = torch.randint(0, 2 * max_shift + 1, (b,), generator=rng_t).to(x.device)
    dx = torch.randint(0, 2 * max_shift + 1, (b,), generator=rng_t).to(x.device)
    ar = torch.arange(SIZE, device=x.device)
    rows, cols = dy[:, None] + ar, dx[:, None] + ar
    return pad[torch.arange(b, device=x.device)[:, None, None], rows[:, :, None], cols[:, None, :]][:, None]


def train_eval(arm, xtr, ytr, xte, yte, seed, steps, bs=64, lr=2e-3, wd=0.05):
    torch.manual_seed(seed)
    model = (SmallCNN() if arm.startswith("cnn") else small_vit()).to(DEVICE)
    xtr, ytr, xte, yte = (t.to(DEVICE) for t in (xtr, ytr, xte, yte))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    n = xtr.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g).to(DEVICE)
        xb, yb = xtr[idx], ytr[idx]
        if arm.endswith("_shift"):
            xb = shift_aug(xb, g)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        acc = (model(xte).argmax(1) == yte).float().mean().item()
        tr_acc = (model(xtr[:2000]).argmax(1) == ytr[:2000]).float().mean().item()
    return acc, tr_acc, sum(p.numel() for p in model.parameters())


def ci95(a):
    a = np.asarray(a, float)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), float(h)


def main():
    quick = "--quick" in sys.argv
    sizes = [60, 200, 600, 2000, 6000, 20000] if not quick else [200]
    seeds = list(range(5)) if not quick else [0]
    steps = 2000 if not quick else 200
    arms = ["cnn", "cnn_shift", "vit", "vit_shift"]
    torch.set_num_threads(4)
    xte, yte = make_images(3000, np.random.default_rng(10_000))  # 시험 세트: 모든 시드가 공유(3000장)
    res = {"sizes": sizes, "seeds": seeds, "steps": steps, "noise": NOISE, "classes": CLASSES,
           "test_n": 3000, "runs": {}}
    t0 = time.time()
    for n in sizes:
        for arm in arms:
            accs, tr = [], []
            for s in seeds:
                xtr, ytr = make_images(n, np.random.default_rng(1000 * s + n))  # 시드마다 다른 학습 세트
                a, ta, npar = train_eval(arm, xtr, ytr, xte, yte, s, steps)
                accs.append(a)
                tr.append(ta)
            res["runs"][f"{arm}|{n}"] = dict(acc=accs, train_acc=tr, params=npar)
            m, h = ci95(accs) if len(accs) > 1 else (accs[0], float("nan"))
            print(f"n={n:5d} {arm:10s} acc {m:.3f} ±{h:.3f}  train {np.mean(tr):.3f}  params {npar}  [{time.time() - t0:.0f}s]",
                  flush=True)
    # 쌍 차이(같은 시드): CNN - ViT, ViT_shift - ViT
    res["diffs"] = {}
    for n in sizes:
        for a, b in [("cnn", "vit"), ("cnn_shift", "vit_shift"), ("vit_shift", "vit"), ("cnn_shift", "cnn"), ("cnn_shift", "vit")]:
            d = np.array(res["runs"][f"{a}|{n}"]["acc"]) - np.array(res["runs"][f"{b}|{n}"]["acc"])
            m, h = ci95(d) if len(d) > 1 else (d[0], float("nan"))
            res["diffs"][f"{a}-{b}|{n}"] = dict(mean=m, ci95=h, per_seed=d.tolist())
    res["seconds"] = time.time() - t0
    if not quick:
        (HERE / "toy_results.json").write_text(json.dumps(res, indent=1))
    print("done", round(res["seconds"]), "s")


if __name__ == "__main__":
    main()
