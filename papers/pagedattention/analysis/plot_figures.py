import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("fragmentation_sim_results.json"))["summary"]
sh = json.load(open("sharing_sim_results.json"))
paper = {"Orca(Max)": (20.4, 13.3, 57.3, 8.9), "Orca(Pow2)": (26.8, 17.9, 13.6, 41.6),
         "Orca(Oracle)": (38.2, 25.2, 0.0, 36.6), "vLLM": (96.3, 0.0, 0.0, 3.7)}
cols = ["#4c9f50", "#e8a838", "#d6404e", "#aaaaaa"]
labs = ["token states", "reservation", "internal frag.", "external frag. & others"]
names = [("max", "Orca(Max)"), ("pow2", "Orca(Pow2)"), ("oracle", "Orca(Oracle)"), ("paged_B16_recompute", "vLLM")]

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, ds in zip(ax, ("sharegpt", "alpaca")):
    for i, (k, nm) in enumerate(names):
        r = d[f"{ds}|cv1.0|{k}"]
        vals = [100 * r[c][0] for c in ("token", "reservation", "internal", "external_others")]
        b = 0
        for v, c in zip(vals, cols):
            a.bar(i * 2, v, bottom=b, color=c, width=0.8, label=labs[cols.index(c)] if i == 0 else None); b += v
        if ds == "sharegpt":
            b = 0
            for v, c in zip(paper[nm], cols):
                a.bar(i * 2 + 0.9, v, bottom=b, color=c, width=0.8, alpha=0.45, hatch="//"); b += v
    a.set_xticks([i * 2 + 0.45 for i in range(4)]); a.set_xticklabels([n for _, n in names], fontsize=8)
    a.set_title(f"{ds}: ours (solid)" + (" vs paper Fig.2 (hatched)" if ds == "sharegpt" else ""), fontsize=9)
ax[0].set_ylabel("KV-cache memory (%)"); ax[0].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("../figures/own_fig2_composition.png", dpi=140); plt.close()

Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for ds, c in (("sharegpt", "C0"), ("alpaca", "C1")):
    t = [d[f"{ds}|cv1.0|paged_B{B}_recompute"]["token"] for B in Bs]
    ax[0].errorbar(Bs, [100 * x[0] for x in t], yerr=[[100 * (x[0] - x[1]) for x in t], [100 * (x[2] - x[0]) for x in t]], marker="o", color=c, label=ds)
    ax[1].plot(Bs, [d[f"{ds}|cv1.0|paged_B{B}_recompute"]["batch"][0] for B in Bs], marker="o", color=c, label=ds)
for a, y in zip(ax, ("token states (% of pool)", "mean running batch (requests)")):
    a.set_xscale("log", base=2); a.set_xlabel("block size B"); a.set_ylabel(y); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("../figures/own_block_size.png", dpi=140); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ks = [2, 4, 6]
pp = {"alpaca": (6.09, 8.53, 9.79), "sharegpt": (16.2, None, 30.5)}
for ds, c in (("alpaca", "C1"), ("sharegpt", "C0")):
    ax[0].errorbar(ks, [sh["parallel"][f"{ds}|n={n}"][0] for n in ks], yerr=[[sh["parallel"][f"{ds}|n={n}"][0] - sh["parallel"][f"{ds}|n={n}"][1] for n in ks], [sh["parallel"][f"{ds}|n={n}"][2] - sh["parallel"][f"{ds}|n={n}"][0] for n in ks]], marker="o", color=c, label=f"{ds} ours")
ax[0].plot(ks, pp["alpaca"], "x--", color="C1", label="alpaca paper Fig.15")
ax[0].plot([2, 6], [16.2, 30.5], "x--", color="C0", label="sharegpt paper (range)")
ax[0].set_xlabel("parallel samples n"); ax[0].set_ylabel("memory saving (%)"); ax[0].set_title("parallel sampling", fontsize=9); ax[0].legend(fontsize=7)
for r, ls in ((-1, ":"), (0.3, "-"), (1.0, "--")):
    ax[1].plot(ks, [sh["beam"][f"alpaca|k={k}|r={r}"][0] for k in ks], ls, marker="o", color="C1", label=f"ours r={r}")
ax[1].plot(ks, (37.56, 53.13, 55.16), "x-", color="k", label="paper Fig.15")
ax[1].set_xlabel("beam width k"); ax[1].set_title("beam search (alpaca)", fontsize=9); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("../figures/own_sharing.png", dpi=140); plt.close()
