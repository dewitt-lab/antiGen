#!/usr/bin/env python3
"""Build spike-restricted (nucleotide) inputs for the runtime comparison.

Restricts to the SARS-CoV-2 spike CDS, NC_045512 / MN908947 positions
21563..25384 (3822 nt), and rebases coordinates to 1..3822 so both tools operate
on the short spike window while still comparing nucleotide placement.

Writes:
  ~/sc2-open-data/ref_spike.json    {"nuc": <3822-nt spike slice>}
  ~/sc2-open-data/tree_spike.json   tree.json with branch_attrs.mutations.nuc
                                     filtered to spike + rebased to 1..3822
"""
import json, re, sys, copy

SPIKE_START, SPIKE_END = 21563, 25384          # 1-based inclusive
OFFSET = SPIKE_START - 1                        # subtract to rebase (21563 -> 1)
DATA = "/home/ivan.specht/sc2-open-data"
MUT = re.compile(r"^([ACGTN-])(\d+)([ACGTN-])$")

ref = json.load(open(f"{DATA}/ref.json"))
nuc = ref["nuc"].upper()
spike = nuc[SPIKE_START-1:SPIKE_END]
assert len(spike) == 3822, len(spike)
json.dump({"nuc": spike}, open(f"{DATA}/ref_spike.json", "w"))
print(f"wrote ref_spike.json ({len(spike)} nt)", file=sys.stderr)

tree_data = json.load(open(f"{DATA}/tree.json"))

kept = dropped = 0
def rebase(node):
    global kept, dropped
    ba = node.get("branch_attrs", {})
    muts = ba.get("mutations", {})
    if "nuc" in muts:
        new = []
        for m in muts["nuc"]:
            mm = MUT.match(m)
            if not mm:
                continue
            frm, pos, to = mm.group(1), int(mm.group(2)), mm.group(3)
            if SPIKE_START <= pos <= SPIKE_END:
                new.append(f"{frm}{pos-OFFSET}{to}")
                kept += 1
            else:
                dropped += 1
        # keep only nuc, rebased (drop any other segment keys)
        node["branch_attrs"]["mutations"] = {"nuc": new}
    for c in node.get("children", []):
        rebase(c)

rebase(tree_data["tree"])
json.dump(tree_data, open(f"{DATA}/tree_spike.json", "w"))
print(f"wrote tree_spike.json (kept {kept} spike nuc muts, dropped {dropped} off-spike)", file=sys.stderr)
