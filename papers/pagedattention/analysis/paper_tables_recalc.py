"""Arithmetic re-checks of numbers printed in the paper (no hardware needed) + a preemption cost toy.

Sources: Table 1, Fig. 1, Fig. 2, Fig. 11, Fig. 13, Fig. 15, Sec. 3 (800 KB/token), Sec. 6.2.
The cost toy (swap vs recompute) uses ASSUMED hardware constants and is not Fig. 19.
"""
import json

GiB = 2 ** 30
out = {}

def kv_bytes(layers, hidden, kv_hidden=None, dtype=2):
    return 2 * (kv_hidden or hidden) * layers * dtype

# --- KV bytes per token (MHA, fp16) and Table 1 slots
models = {"OPT-13B": (40, 5120), "OPT-66B": (64, 9216), "OPT-175B": (96, 12288)}
table1 = {"OPT-13B": (12, 15.7e3), "OPT-66B": (21, 9.7e3), "OPT-175B": (264, 60.1e3)}
rows = {}
for m, (L, h) in models.items():
    b = kv_bytes(L, h)
    gb, slots = table1[m]
    rows[m] = {"bytes_per_token": b, "KiB_per_token": b / 1024,
               "slots_if_GiB": gb * GiB / b, "slots_if_1e9": gb * 1e9 / b, "paper_slots": slots,
               "GiB_for_paper_slots": slots * b / GiB, "bytes_2048_tokens_GiB": 2048 * b / GiB,
               "bytes_2048_tokens_GB": 2048 * b / 1e9}
out["table1_and_kv_per_token"] = rows
out["other_models_bytes_per_token"] = {
    "Llama-2-7B (MHA 32L x 4096)": kv_bytes(32, 4096),
    "Llama-3-8B (GQA 8 kv heads x 128, 32L)": kv_bytes(32, 4096, kv_hidden=1024)}

# --- Fig. 1 / Table 1 consistency for OPT-13B on A100-40GB
out["fig1"] = {"weights_GB_fp16": 13e9 * 2 / 1e9, "weights_GiB": 13e9 * 2 / GiB,
               "weights_share_of_40GB": 26 / 40, "KV_12GiB_share_of_40GiB": 12 / 40}

# --- Fig. 2 stacked bars (read from the figure): row sums and the 'external' of Orca(Max)
fig2 = {"Orca(Max)": (20.4, 13.3, 57.3, 8.9), "Orca(Pow2)": (26.8, 17.9, 13.6, 41.6),
        "Orca(Oracle)": (38.2, 25.2, 0.0, 36.6), "vLLM": (96.3, None, None, None)}
out["fig2_row_sums"] = {k: (sum(v) if None not in v else None) for k, v in fig2.items()}
slots = 15728
out["orca_max_external_recomputed_pct"] = 100 * (slots - (slots // 2048) * 2048) / slots  # 7 x 2048 fit

# --- Fig. 11 ratios and Fig. 13 ratios
out["fig11_ratios"] = {"input_mean_ratio": 161.31 / 19.31, "output_mean_ratio": 337.99 / 58.45,
                       "paper_text": (8.4, 5.8)}
f13s, f13a = (7.00, 9.81, 13.62, 30.42), (7.00, 43.24, 72.75, 132.44)
out["fig13_ratios"] = {"sharegpt_vllm_over_oracle": f13s[3] / f13s[2], "sharegpt_vllm_over_max": f13s[3] / f13s[0],
                       "paper_text": (2.2, 4.3), "alpaca_vllm_over_oracle": f13a[3] / f13a[2],
                       "alpaca_vllm_over_max": f13a[3] / f13a[0],
                       "orca_max_batch_bound": slots // 2048}

# --- Fig. 15 values (read from figure) vs text ranges
out["fig15"] = {"alpaca_parallel_n246": (6.09, 8.53, 9.79), "alpaca_beam_k246": (37.56, 53.13, 55.16),
                "text": "6.1-9.8 parallel, 37.6-55.2 beam (Alpaca); 16.2-30.5, 44.3-66.3 (ShareGPT)"}

# --- Sec. 6.3 text: prompt part is 12% of KV in the parallel-sampling experiment (no recompute possible)
# --- preemption cost toy (ASSUMED constants)
L, h = 40, 5120
bytes_tok = kv_bytes(L, h)
PCIE = 24e9           # B/s effective, assumption
T_XFER = 8e-6         # s per cudaMemcpy call (one per block per layer per K/V), assumption
FLOPS = 0.4 * 312e12  # 40% of A100 fp16 peak, assumption
P = 13e9
toy = {}
for B in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    r = {}
    for n in (64, 512, 2048):
        blocks = -(-n // B)
        swap = 2 * (n * bytes_tok / PCIE + blocks * 2 * L * T_XFER)   # out + in
        recompute = 2 * P * n / FLOPS
        r[str(n)] = {"swap_ms": 1e3 * swap, "recompute_ms": 1e3 * recompute}
    toy[f"B={B}"] = r
out["preemption_cost_toy"] = {"assumptions": {"PCIe_B_per_s": PCIE, "per_copy_s": T_XFER,
                              "flops": FLOPS, "params": P}, "ms": toy}
json.dump(out, open("paper_tables_recalc_results.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in out if k != "preemption_cost_toy"}, indent=1))
for B in (1, 16, 256):
    print(B, toy[f"B={B}"]["512"])
