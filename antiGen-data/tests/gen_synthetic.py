#!/usr/bin/env python3
"""Generate a deterministic synthetic AA dataset for the CPU/GPU parity test.

Produces a Nextstrain-style tree.json (multi-level guide tree), ref.json, seqs.fasta,
and metadata.tsv under <out_dir>, using the `toy` dataset type's protein alphabet and
its fixed length (L=25). The attach set mixes exact copies of the guide genotypes with
0-4 random mutations so the attachment exercises all six wiring cases.

Fixed RNG seed => byte-identical inputs on every run, so a baseline captured once can be
compared against later runs.

Usage: gen_synthetic.py <out_dir>
"""
import json
import random
import sys
import datetime

L = 25  # must match the `toy` dataset type's total_length in create_dataset_config
AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical residues (no ambiguity char)
rng = random.Random(20240714)


def mutate(seq, n):
    s = list(seq)
    muts = []
    for p in sorted(rng.sample(range(L), n)):
        old = s[p]
        new = rng.choice([a for a in AA if a != old])
        s[p] = new
        muts.append(f"{old}{p + 1}{new}")  # 1-indexed <ref><pos><alt>
    return "".join(s), muts


def main():
    out = sys.argv[1]
    import os
    os.makedirs(out, exist_ok=True)

    root = "".join(rng.choice(AA) for _ in range(L))
    a_seq, a_muts = mutate(root, 3)
    b_seq, b_muts = mutate(a_seq, 2)
    c_seq, c_muts = mutate(root, 4)

    tree = {
        "version": "v2",
        "meta": {"title": "synthetic CPU/GPU parity test (AA, L=25)"},
        "tree": {
            "name": "NODE_ROOT",
            "node_attrs": {"num_date": {"value": 2000.0}, "clade_membership": {"value": "root"}},
            "children": [
                {"name": "A", "node_attrs": {"num_date": {"value": 2000.5}},
                 "branch_attrs": {"mutations": {"protein": a_muts}},
                 "children": [
                     {"name": "B", "node_attrs": {"num_date": {"value": 2001.0}},
                      "branch_attrs": {"mutations": {"protein": b_muts}}},
                 ]},
                {"name": "C", "node_attrs": {"num_date": {"value": 2000.7}},
                 "branch_attrs": {"mutations": {"protein": c_muts}}},
            ],
        },
    }

    genotypes = {"root": root, "A": a_seq, "B": b_seq, "C": c_seq}
    records = []  # (name, seq)
    for gname, g in genotypes.items():
        for _ in range(5):
            records.append((f"exact_{gname}_{len(records)}", g))
    for _ in range(300):
        g = genotypes[rng.choice(list(genotypes))]
        s, _ = mutate(g, rng.randint(0, 4))
        records.append((f"mut_{len(records)}", s))

    with open(f"{out}/ref.json", "w") as f:
        json.dump({"protein": root}, f)
    with open(f"{out}/tree.json", "w") as f:
        json.dump(tree, f, indent=2)
    with open(f"{out}/seqs.fasta", "w") as f:
        for name, s in records:
            f.write(f">{name}\n{s}\n")
    base = datetime.date(2002, 1, 1)
    with open(f"{out}/metadata.tsv", "w") as f:
        f.write("strain\tdate\n")
        for i, (name, _) in enumerate(records):
            d = base + datetime.timedelta(days=i)
            f.write(f"{name}\t{d.isoformat()}\n")

    print(f"wrote {len(records)} attach sequences (L={L}) to {out}/")


if __name__ == "__main__":
    main()
