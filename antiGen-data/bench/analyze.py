#!/usr/bin/env python3
"""Analyze bench/results.csv: mine UShER logs for VCF-load time (to report a fair
'core placement' metric alongside total wall), print a summary table, and render
a log-log scaling-curve plot (time vs N).

Metrics:
  antiGen-data  core_ms  = attach-loop wall clock (excludes tree parse + FASTA read)
  usher-*       wall_s   = full invocation (guide .pb prebuilt)
  usher-*       place_s  = wall_s - VCF-load time (mined from log) ~ pure placement
                           (usher-sampled 'core_ms' = its own 'Took' timer, place+optimize)
"""
import csv, re, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BENCH, "results.csv")

def vcf_load_s(logpath):
    """Extract 'Loading VCF file ... Completed in X msec' -> seconds, else None."""
    try:
        txt = open(logpath, errors="ignore").read()
    except FileNotFoundError:
        return None
    m = re.search(r"Loading VCF file\s*\nCompleted in (\d+) msec", txt)
    if m:
        return int(m.group(1)) / 1000.0
    return None

rows = []
# main sweep CSV plus any per-job 1M result files (written by separate jobs)
csv_files = [CSV] + [os.path.join(BENCH, f) for f in
                     ("results_1M_antigen.csv", "results_1M_ushs.csv", "results_1M_ushc.csv")]
for path in csv_files:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

def num(x):
    try: return float(x)
    except: return None

# Build a tidy record per row with a chosen "augmentation seconds" value.
recs = []
for r in rows:
    tool, mode, N = r["tool"], r["mode"], int(r["N"])
    wall = num(r["wall_s"]); core_ms = num(r["core_ms"]); rss = num(r["max_rss_kb"])
    logmap = {"usher-classic": f"ushc_{N}.log", "usher-sampled": f"ushs_{N}.log"}
    load = vcf_load_s(os.path.join(BENCH, "data", logmap[tool])) if tool in logmap else None
    place_s = (wall - load) if (wall is not None and load is not None) else None
    if tool == "antiGen-data":
        aug_s = core_ms / 1000.0 if core_ms is not None else wall  # attach loop
        metric = "attach-loop"
    elif tool == "usher-sampled":
        # its own "Took" timer (place+optimize), excludes VCF load; for the 1M
        # batched run core_ms is the summed per-batch placement time.
        aug_s = core_ms / 1000.0 if core_ms else (place_s if place_s is not None else wall)
        metric = "place+optimize (Took)"
    else:
        aug_s = place_s if place_s is not None else wall          # placement (load-excluded)
        metric = "placement (wall-load)"
    recs.append(dict(tool=tool, mode=mode, N=N, wall_s=wall, core_ms=core_ms,
                     vcf_load_s=load, aug_s=aug_s, metric=metric, rss_mb=(rss/1024 if rss else None)))

# ---- table ----
recs.sort(key=lambda x: (x["N"], x["tool"]))
print(f"{'tool':14} {'mode':16} {'N':>8} {'wall_s':>9} {'vcf_load_s':>11} {'aug_s':>10} {'rss_MB':>8}")
for r in recs:
    print(f"{r['tool']:14} {r['mode']:16} {r['N']:>8} "
          f"{(r['wall_s'] if r['wall_s'] is not None else float('nan')):>9.2f} "
          f"{(r['vcf_load_s'] if r['vcf_load_s'] is not None else float('nan')):>11.2f} "
          f"{(r['aug_s'] if r['aug_s'] is not None else float('nan')):>10.3f} "
          f"{(r['rss_mb'] if r['rss_mb'] is not None else float('nan')):>8.0f}")

# ---- plot: augmentation seconds vs N (log-log) ----
tools = {}
for r in recs:
    if r["aug_s"] is None: continue
    tools.setdefault(r["tool"], []).append((r["N"], r["aug_s"]))

style = {
    "antiGen-data":  dict(color="#2563eb", marker="o", label="antiGen-data (GPU/H100, attach loop)"),
    "usher-sampled": dict(color="#f59e0b", marker="s", label="usher-sampled (CPU, place+optimize)"),
    "usher-classic": dict(color="#dc2626", marker="^", label="usher classic (CPU, placement)"),
}
plt.figure(figsize=(7.5, 5.2))
for tool, pts in tools.items():
    pts.sort()
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    st = style.get(tool, dict(marker="x", label=tool))
    plt.plot(xs, ys, linewidth=2, markersize=7, **st)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Number of appended samples (N)")
plt.ylabel("Augmentation time (seconds)")
plt.title("Tree augmentation: antiGen-data vs UShER\nSARS-CoV-2 whole-genome, guide = tree.json (4,549 tips)")
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
out = os.path.join(BENCH, "scaling.png")
plt.savefig(out, dpi=150)
print(f"\nwrote {out}")
