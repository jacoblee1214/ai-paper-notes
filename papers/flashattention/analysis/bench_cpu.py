"""CPU에서 attention 구현 세 가지의 시간과 최대 상주 메모리(RSS)를 잰다. 예비 실험이며 논문의 GPU 수치와 직접 비교하지 않는다.

  naive : S, P를 만들어 저장하는 표준 구현 (torch.nn.attention.SDPBackend.MATH와 같은 수식)
  sdpa  : PyTorch의 CPU용 타일 attention 커널(SDPBackend.FLASH_ATTENTION, 공식 FlashAttention 코드가 아니라 PyTorch 자체 CPU 구현)
  tiled : 이 폴더 minimal_impl.flash_forward (파이썬 루프, 순전파만)
설정마다 새 프로세스를 띄워 RSS 증가분(입력 q, k, v를 만든 직후 대비 최댓값)을 잰다. 헤드 2개, d=64, float32, 스레드 4개. 시간은 5회 중 중앙값.
실행: python3 bench_cpu.py   (결과 bench_cpu_results.json)
"""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

CHILD = r"""
import json, resource, sys, time, math, torch
sys.path.insert(0, sys.argv[5])
import minimal_impl as mi
impl, N, H, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
torch.set_num_threads(4); torch.manual_seed(0)
d = 64
q, k, v = (torch.randn(1, H, N, d, requires_grad=(mode == "fwdbwd" and impl != "tiled")) for _ in range(3))
from torch.nn.attention import SDPBackend, sdpa_kernel
def run():
    if impl == "naive":
        S = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        o = torch.softmax(S, -1) @ v
    elif impl == "sdpa":
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        with torch.no_grad():
            o = torch.stack([mi.flash_forward(q[0, h], k[0, h], v[0, h], 98304)[0] for h in range(H)])[None]
    if mode == "fwdbwd" and impl != "tiled":
        o.sum().backward()
    return o
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
run()  # warm up (최대 RSS는 첫 실행에서 이미 정해지므로 기준은 그 전에 잰다)
ts = []
for _ in range(5):
    t = time.perf_counter(); run(); ts.append(time.perf_counter() - t)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
div = 1 if sys.platform != "darwin" else 1024  # macOS는 바이트, Linux는 KB
print(json.dumps(dict(impl=impl, N=N, H=H, mode=mode, ms_median=1000 * sorted(ts)[2], rss_mb_delta=(peak - base) / div / 1024)))
"""


def main():
    here = Path(__file__).parent
    rows = []
    for mode in ("fwd", "fwdbwd"):
        for N in (1024, 2048, 4096, 8192):
            for impl in ("naive", "sdpa", "tiled"):
                if impl == "tiled" and mode == "fwdbwd":
                    continue
                out = subprocess.run([sys.executable, "-c", CHILD, impl, str(N), "2", mode, str(here)], capture_output=True, text=True, timeout=600)
                if out.returncode != 0:
                    print(impl, N, mode, "FAIL", out.stderr[-300:]); continue
                rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
                print(rows[-1], flush=True)
    (here / "bench_cpu_results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
