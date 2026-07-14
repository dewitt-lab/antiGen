#!/usr/bin/env python3
"""Select a deterministic, date-valid, spread-out master list of append samples
from the UCSC public MAT sample list. Writes append_master.txt (sample names,
one per line) and size-prefixed lists append_<N>.txt for the benchmark sweep.

Sample names look like: strain|accession|YYYY-MM-DD  -> require a full date.
Deterministic ordering via a seeded hash of the name (reproducible, spread).
"""
import sys, re, hashlib, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default="bench/data/public_samples.txt")
ap.add_argument("--outdir", default="bench/data")
ap.add_argument("--max", type=int, default=1_000_000)
ap.add_argument("--sizes", default="1000,10000,100000,500000,1000000")
ap.add_argument("--seed", default="antigen-bench-42")
args = ap.parse_args()

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
seed = args.seed.encode()

# Collect (hashkey, name) for date-valid samples.
scored = []
n_total = 0
with open(args.samples) as f:
    header = f.readline()  # skip 'sample\tparsimony\tparent_id'
    for line in f:
        n_total += 1
        name = line.split("\t", 1)[0]
        parts = name.split("|")
        if len(parts) < 3 or not DATE_RE.match(parts[-1]):
            continue
        h = hashlib.md5(seed + name.encode()).digest()
        scored.append((h, name))

print(f"scanned {n_total} samples; {len(scored)} have a full date", file=sys.stderr)
scored.sort(key=lambda x: x[0])
chosen = [name for _, name in scored[: args.max]]

master = f"{args.outdir}/append_master.txt"
with open(master, "w") as f:
    f.write("\n".join(chosen) + "\n")
print(f"wrote {master} ({len(chosen)} names)", file=sys.stderr)

for s in args.sizes.split(","):
    n = int(s)
    if n > len(chosen):
        print(f"  skip N={n} (only {len(chosen)} available)", file=sys.stderr)
        continue
    p = f"{args.outdir}/append_{n}.txt"
    with open(p, "w") as f:
        f.write("\n".join(chosen[:n]) + "\n")
    print(f"  wrote {p}", file=sys.stderr)
