#!/usr/bin/env python3
"""Overlay whole-genome vs spike-restricted augmentation runtimes (log-log).
Reads results_all.csv (whole-genome) + results_spike.csv; writes scaling_spike.png.
Aug time = core_ms/1000 (antiGen attach loop; usher-sampled 'Took' timer)."""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.path.dirname(os.path.abspath(__file__))

def load(path):
    d = {}
    if not os.path.exists(path): return d
    for r in csv.DictReader(open(path)):
        try: core = float(r["core_ms"])
        except: continue
        d.setdefault(r["tool"], []).append((int(r["N"]), core/1000.0))
    for k in d: d[k].sort()
    return d

wg, sp = load(f"{B}/results_all.csv"), load(f"{B}/results_spike.csv")
# usher-classic core_ms is NA (placement-only, no self-timer); use wall_s instead.
def load_classic(path):
    pts = []
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["tool"] == "usher-classic":
                try: pts.append((int(r["N"]), float(r["wall_s"])))
                except: pass
    return sorted(pts)
if load_classic(f"{B}/results_spike.csv"): sp["usher-classic"] = load_classic(f"{B}/results_spike.csv")
if load_classic(f"{B}/results_all.csv"):   wg["usher-classic"] = load_classic(f"{B}/results_all.csv")

col = {"antiGen-data": "#2563eb", "usher-sampled": "#f59e0b", "usher-classic": "#dc2626"}
lbl = {"antiGen-data": "antiGen-data (GPU)", "usher-sampled": "usher-sampled (CPU)",
       "usher-classic": "usher classic, Turakhia et al. (CPU)"}

plt.figure(figsize=(8, 5.6))
for tool in ("antiGen-data", "usher-sampled", "usher-classic"):
    if tool in wg:
        xs=[p[0] for p in wg[tool]]; ys=[p[1] for p in wg[tool]]
        plt.plot(xs, ys, "--", color=col[tool], marker="o", ms=6, alpha=0.6,
                 label=f"{lbl[tool]} — whole genome (29,903 nt)")
    if tool in sp:
        xs=[p[0] for p in sp[tool]]; ys=[p[1] for p in sp[tool]]
        plt.plot(xs, ys, "-", color=col[tool], marker="s", ms=7,
                 label=f"{lbl[tool]} — spike only (3,822 nt)")

plt.xscale("log"); plt.yscale("log")
plt.xlabel("Number of appended samples (N)")
plt.ylabel("Augmentation time (seconds)")
plt.title("Tree augmentation: whole-genome vs spike-restricted (nucleotide)\n"
          "SARS-CoV-2, guide = tree.json (4,549 tips)")
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=8.5)
plt.tight_layout()
out = f"{B}/scaling_spike.png"
plt.savefig(out, dpi=150)
print("wrote", out)
