#!/usr/bin/env python3
"""Convert the Nextstrain Auspice tree.json (with nuc mutations) into a Newick
tree + a per-leaf VCF, so UShER can build a mutation-annotated guide tree (.pb)
from the SAME phylogeny that antiGen-data consumes directly.

Reconstructs each leaf's genotype by accumulating branch_attrs.mutations.nuc
along the root->leaf path (relative to MN908947 = ref.json["nuc"]).
Emits a VCF with one column per leaf; only variable A/C/G/T sites are written.
Gaps ('-') and ambiguous ('N') alleles are written as missing ('.') — UShER
ignores indels/ambiguities anyway.
"""
import json, sys, re, argparse

sys.setrecursionlimit(100000)

ap = argparse.ArgumentParser()
ap.add_argument("--tree", default="/home/ivan.specht/sc2-open-data/tree.json")
ap.add_argument("--ref", default="/home/ivan.specht/sc2-open-data/ref.json")
ap.add_argument("--out-nwk", default="bench/guide.nwk")
ap.add_argument("--out-vcf", default="bench/guide.vcf")
ap.add_argument("--ref-name", default="MN908947.3")
args = ap.parse_args()

ref = json.load(open(args.ref))["nuc"].upper()
tree = json.load(open(args.tree))["tree"]

MUT_RE = re.compile(r"^([ACGTN-])(\d+)([ACGTN-])$")

# --- Newick emission (leaf names only; no branch lengths needed by usher) ---
def to_newick(node):
    children = node.get("children", [])
    name = node.get("name", "")
    if not children:
        return name
    inner = ",".join(to_newick(c) for c in children)
    return f"({inner})"

# --- Accumulate per-leaf genotypes (sparse: pos -> allele) ---
# leaf_variants: list of (leaf_name, {pos(1-based): allele})
leaf_variants = []
variable_positions = set()

def walk(node, geno):
    muts = node.get("branch_attrs", {}).get("mutations", {}).get("nuc", [])
    local = None
    for m in muts:
        mm = MUT_RE.match(m)
        if not mm:
            continue
        frm, pos, to = mm.group(1), int(mm.group(2)), mm.group(3)
        if local is None:
            local = dict(geno)  # copy only when this branch has mutations
        local[pos] = to
    cur = local if local is not None else geno
    children = node.get("children", [])
    if not children:
        # leaf: record variants that differ from reference
        variants = {}
        for pos, allele in cur.items():
            if pos < 1 or pos > len(ref):
                continue
            if allele != ref[pos - 1]:
                variants[pos] = allele
                variable_positions.add(pos)
        leaf_variants.append((node.get("name", ""), variants))
    else:
        for c in children:
            walk(c, cur)

walk(tree, {})

leaf_names = [n for n, _ in leaf_variants]
print(f"leaves: {len(leaf_names)}, variable positions: {len(variable_positions)}", file=sys.stderr)

# --- Build VCF ---
with open(args.out_nwk, "w") as f:
    f.write(to_newick(tree) + ";\n")

positions = sorted(variable_positions)
with open(args.out_vcf, "w") as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(leaf_names) + "\n")
    # index leaf variant dicts for fast column lookup
    per_leaf = [v for _, v in leaf_variants]
    for pos in positions:
        refb = ref[pos - 1]
        # collect ALT alleles (A/C/G/T only) observed at this pos
        alts = []
        alt_idx = {}
        gts = []
        for v in per_leaf:
            a = v.get(pos)
            if a is None:
                gts.append("0")            # matches reference
            elif a in "ACGT" and a != refb:
                if a not in alt_idx:
                    alts.append(a)
                    alt_idx[a] = len(alts)  # 1-based ALT index
                gts.append(str(alt_idx[a]))
            elif a == refb:
                gts.append("0")
            else:  # '-' or 'N' or anything ambiguous
                gts.append(".")
        if not alts:
            continue
        f.write(f"{args.ref_name}\t{pos}\t.\t{refb}\t{','.join(alts)}\t.\tPASS\t.\tGT\t" + "\t".join(gts) + "\n")

print(f"wrote {args.out_nwk} and {args.out_vcf} ({len(positions)} candidate sites)", file=sys.stderr)
