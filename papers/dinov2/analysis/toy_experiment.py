"""합성 도형 토이: 중심화와 날카롭게 하기(교사 온도), Sinkhorn-Knopp, KoLeo 가 붕괴를 막는지 확인한다.
논문 재현이 아니다(작은 ViT 0.1M 파라미터, 32x32 합성 이미지, 프로토타입 64개, 600 스텝). 논문 표 1·3 을 대신하지 않는다.

조건 7가지 x 시드 5개. 측정:
  teacher_ent   교사 확률의 표본별 엔트로피 평균(nats). log K = 4.16 이면 균등(붕괴 B), 0 이면 원-핫
  marginal_ent  배치 평균 확률의 엔트로피. 0 에 가까우면 모든 표본이 같은 프로토타입(붕괴 A), log K 면 프로토타입을 고르게 씀
  max_marginal  가장 많이 쓰인 프로토타입의 질량
  erank         교사 [CLS] 특징(평가 이미지 2000장)의 유효 차원 exp(H(s/sum s)) (s: 중심화한 특징의 특이값)
  probe_acc     동결 교사 특징 위의 선형 분류(도형 6종) 정확도. 무작위 초기화 백본의 값도 함께 기록(기준선)
실행: python3 toy_experiment.py [--seeds 5] [--steps 600] [--pilot]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as M  # noqa: E402

K = 64
CONDITIONS = {
    #  이름: (중심화, 교사 온도 방식, KoLeo 가중치)
    "neither":     ("none",   "flat",  0.0),   # 중심화 없음 + 날카롭게 하기 없음(교사 온도 = 학생 온도 0.1)
    "no_center":   ("none",   "sharp", 0.0),   # 날카롭게 하기만
    "no_sharpen":  ("center", "flat",  0.0),   # 중심화만
    "dino":        ("center", "sharp", 0.0),   # DINO: 둘 다
    "sk":          ("sk",     "sharp", 0.0),   # SK 중심화 + 날카롭게 하기
    "dino_koleo":  ("center", "sharp", 0.1),
    "dinov2":      ("sk",     "sharp", 0.1),   # DINOv2: SK + KoLeo
}


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def erank(f):
    f = f - f.mean(0, keepdim=True)
    s = torch.linalg.svdvals(f.float())
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def probe(train_f, train_y, test_f, test_y, steps=300):
    mu, sd = train_f.mean(0, keepdim=True), train_f.std(0, keepdim=True) + 1e-6
    tr, te = (train_f - mu) / sd, (test_f - mu) / sd
    torch.manual_seed(0)
    lin = torch.nn.Linear(tr.shape[1], len(M.SHAPES)).to(tr.device)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(tr), train_y).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(te).argmax(1) == test_y).float().mean())


@torch.no_grad()
def features(model, x):
    return torch.cat([model.teacher["backbone"](x[i:i + 500])[0] for i in range(0, len(x), 500)])


def run(cond, seed, steps, batch, dev, n_local=4, hp=None):
    hp = {**dict(lr=1e-3, ema=0.99, t_start=0.04, t_end=0.07, t_flat=0.1, warm_frac=0.3), **(hp or {})}
    center_mode, temp_mode, koleo_w = CONDITIONS[cond]
    torch.manual_seed(seed)
    model = M.DINOv2Toy(n_proto=K).to(dev)
    pool_x, _ = M.make_shapes(6000, device=dev)
    ptr_x, ptr_y = M.make_shapes(3000, device=dev)
    pte_x, pte_y = M.make_shapes(1500, device=dev)
    ev_x, _ = M.make_shapes(2000, device=dev)

    f0 = features(model, ptr_x), features(model, pte_x)
    init_acc = probe(f0[0], ptr_y, f0[1], pte_y)
    init_erank = erank(features(model, ev_x))

    opt = torch.optim.AdamW(model.student.parameters(), lr=hp['lr'], weight_decay=0.04)
    warm = 40
    log = []
    for it in range(steps):
        lr = hp['lr'] * (it + 1) / warm if it < warm else hp['lr'] * 0.5 * (1 + math.cos(math.pi * (it - warm) / (steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr
        t_temp = hp["t_flat"] if temp_mode == "flat" else M.teacher_temp_schedule(it, int(hp["warm_frac"] * steps), hp["t_start"], hp["t_end"])
        m = M.cosine_schedule(hp['ema'], 1.0, it, steps)
        img = pool_x[torch.randint(0, len(pool_x), (batch,), device=dev)]
        gc = torch.stack([M.color_augment(M.random_resized_crop(img, 32, (0.32, 1.0))) for _ in range(2)])
        lc = torch.stack([M.color_augment(M.random_resized_crop(img, 16, (0.05, 0.32))) for _ in range(n_local)])
        mk = torch.stack([M.random_patch_masks(batch, 64, device=dev) for _ in range(2)])
        loss, d = model.loss(gc, lc, mk, t_temp, centering=center_mode, koleo_weight=koleo_w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3.0)
        opt.step()
        model.update_teacher(m)
        with torch.no_grad():
            p = d  # 진단은 loss 안에서 이미 계산됨
        log.append((d["teacher_ent"], d["marginal_ent"], d["dino"], d["ibot"]))
    tail = torch.tensor(log[-50:]).mean(0).tolist()
    ftr, fte = features(model, ptr_x), features(model, pte_x)
    # 최대 프로토타입 질량: 마지막 상태에서 평가 이미지에 대한 교사 확률(해당 조건의 처리 방식, 온도 0.07)로 잰다
    with torch.no_grad():
        z = model.teacher["dino_head"](features(model, ev_x))
        if center_mode == "center":
            pe = model.dino_center.probs(z, 0.07)
        elif center_mode == "sk":
            pe = M.sinkhorn_knopp(z, 0.07)
        else:
            pe = torch.softmax(z / 0.07, dim=-1)
        marg = pe.mean(0)
    return dict(cond=cond, seed=seed, steps=steps,
                teacher_ent=tail[0], marginal_ent=tail[1], dino_loss=tail[2], ibot_loss=tail[3],
                max_marginal=float(marg.max()), erank=erank(features(model, ev_x)),
                probe_acc=probe(ftr, ptr_y, fte, pte_y), init_probe_acc=init_acc, init_erank=init_erank,
                traj=[(i, log[i][0], log[i][1]) for i in range(0, steps, 25)])


SETTINGS = {
    # 주 설정: 교사 온도 0.02 고정(시드 0 예비 실험에서 공식 0.04->0.07이 균등 붕괴를 낳아 낮춤, 03 §5)
    "main": (dict(t_start=0.02, t_end=0.02), list(CONDITIONS)),
    # 공식 온도 0.04->0.07 (워밍업 30%): 날카롭게 하는 4개 조건만
    "official_temp": (dict(), ["dino", "sk", "dino_koleo", "dinov2"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "toy_results.json"))
    a = ap.parse_args()
    dev = pick_device()
    runs = []
    t0 = time.time()
    for setting, (hp, conds) in SETTINGS.items():
        for seed in range(a.seeds):
            for cond in conds:
                r = run(cond, seed, a.steps, a.batch, dev, hp=hp)
                r["setting"] = setting
                runs.append(r)
                print(f"[{time.time() - t0:6.0f}s] {setting} seed {seed} {cond:11s} ent {r['teacher_ent']:.2f} marg {r['marginal_ent']:.2f} "
                      f"max {r['max_marginal']:.2f} erank {r['erank']:.1f} probe {r['probe_acc']:.3f} (init {r['init_probe_acc']:.3f})", flush=True)
                Path(a.out).write_text(json.dumps(dict(device=dev, K=K, log_K=math.log(K), steps=a.steps, batch=a.batch,
                                                        conditions=CONDITIONS, settings={k: v[0] for k, v in SETTINGS.items()},
                                                        runs=runs), indent=1))


if __name__ == "__main__":
    main()
