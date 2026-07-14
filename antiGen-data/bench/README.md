# Benchmark: antiGen-data vs. UShER (tree-augmentation runtime)

This directory contains everything needed to reproduce the runtime comparison of
**antiGen-data** (GPU) against **UShER** (CPU) on the *tree-augmentation* phase —
placing N new samples onto a fixed guide tree — using SARS-CoV-2 whole-genome
nucleotide data.

UShER is nucleotide-only, so the comparison uses antiGen-data's `sc2_nuc` dataset
mode (whole-genome SARS-CoV-2, Wuhan-Hu-1 / MN908947, 29,903 nt) added for this
purpose. Both tools start from the **same guide tree** and place the **same
append samples**, derived from a single source so the comparison is like-for-like.

## Data provenance

- **Guide tree:** `~/sc2-open-data/tree.json` — a Nextstrain Auspice v2 SARS-CoV-2
  build, 4,549 tips / 8,859 nodes, carrying `branch_attrs.mutations.nuc` (full-genome
  coordinates) and `node_attrs.num_date` on every node.
- **Reference:** `~/sc2-open-data/ref.json` → `["nuc"]` = Wuhan-Hu-1 (GenBank
  **MN908947.3** / RefSeq NC_045512v2), 29,903 nt.
- **Append samples:** the UCSC public SARS-CoV-2 mutation-annotated tree
  (`public-latest.all.masked.pb`, downloaded 2026-07-03 from
  `hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/`), build version
  *"8,454,907 genomes from GenBank, COG-UK and CNCB (2026-07-02)"*. 8,454,908
  samples; SNVs relative to MN908947, problematic sites masked to `N`, indels
  excluded. 8,368,973 have a full `YYYY-MM-DD` collection date.

## Pipeline

1. **`select_samples.py`** — deterministically (seeded md5 hash) select a
   date-valid, phylogenetically dispersed master list of 1,000,000 append samples;
   nested prefixes give the sweep sizes (`append_{1000,10000,100000,1000000}.txt`).
2. **`prep.sh <N>`** — `matUtils extract` a per-sample SNV VCF for those samples
   (UShER's placement input), then **`reconstruct_fasta.py`** applies each sample's
   ALT alleles to MN908947 to emit a reference-length (29,903 nt) FASTA + a
   `strain,date` metadata TSV (antiGen-data's input). Both tools thus receive
   identical, identically-aligned samples from one VCF. Missing/masked calls -> `N`.
3. **`tree_to_vcf.py`** — convert `tree.json` -> Newick + per-tip VCF; then
   `usher --tree guide.nwk --vcf guide.vcf --collapse-tree --save-mutation-annotated-tree guide.pb`
   builds the UShER guide `.pb` from the SAME phylogeny antiGen-data consumes directly.
4. **`bench_one.sh <N> <threads>`** / **`run_tool.sh <tool> <N> <threads> <csv>`** —
   run the augmentation phase and record timings to CSV. Tools:
   - `antiGen-data` (GPU, `sc2_nuc`, `ANTIGEN_BENCH_ATTACH_ONLY=1` times just the
     attach loop via `AUGMENTATION_WALL_CLOCK_MS`);
   - `usher-sampled` (placement + SPR refinement; "most efficient" mode);
   - classic `usher` (placement only; strict algorithmic analog, small N only).
5. **`analyze.py`** — merge CSVs, mine UShER logs for VCF-load time, print a table,
   render `scaling.png` (log-log time vs N).

### Note on the N=1M UShER point

`usher-sampled` **crashes placing 1,000,000 samples in a single invocation**
(`node id Mismatch` → segfault, ~9 GB RSS, not OOM); it succeeds at ≤100k. The 1M
point is therefore obtained by **incremental placement** — 10 sequential batches of
100k onto the growing MAT (`slurm/run_ushs_1M_batched.slurm`), which is how UShER is used
operationally. Only the placement invocations are timed (VCF extraction is prep).
antiGen-data places all 1M in one run. The single-batch crash is itself a
robustness finding worth reporting.

### SLURM wrappers (in `slurm/` at the repo root)

- `slurm/bench.slurm <N...>` — sweep on one GPU node (antiGen on GPU + UShER on its CPUs).
- `slurm/prep_1M.slurm`, `slurm/run_antigen_1M.slurm`, `slurm/run_ushs_1M.slurm` — the
  N=1M point as dependent jobs (prep -> antiGen ∥ usher-sampled), so the H100 isn't
  held during UShER's long CPU run.
- `slurm/smoke.slurm` — tiny end-to-end GPU sanity check.

Run them from the repo root (each script `cd`s there and uses `bench/`-relative paths).

## What each tool measures

- **antiGen-data:** GPU attach-loop wall clock (`AUGMENTATION_WALL_CLOCK_MS`) —
  excludes tree parse, guide reconstruction, and FASTA read.
- **UShER:** full invocation wall clock with the guide `.pb` prebuilt; `analyze.py`
  subtracts VCF-load time to approximate pure placement. `usher-sampled` also
  reports its own "Took" timer (place+optimize).

**Caveat (state in any writeup):** the algorithms differ — UShER does optimal
Fitch-Sankoff parsimony over sparse variants (usher-sampled additionally runs SPR
refinement); antiGen-data does a Hamming-to-both-endpoints heuristic over the dense
alignment, scoring each new sample against every current edge on the GPU. This is a
fair *time-to-augment* comparison, not a claim of identical placements.

## Environment

- antiGen-data: one NVIDIA H100 80 GB; CUDA 12.6; built with `make` (see top-level).
- UShER v0.6.6 (`usher`, `usher-sampled`, `matUtils`, `faToVcf`; bioconda,
  `conda activate usher`). Reconstruction uses base-conda Python + numpy.

## Results (1k → 1M)

Core augmentation time (antiGen GPU attach loop on one H100; UShER 16 CPU threads):

| N         | antiGen-data | usher-sampled       | classic usher | antiGen speedup |
|-----------|--------------|---------------------|---------------|-----------------|
| 1,000     | 0.43 s       | 2.08 s              | 16.4 s        | 4.8×            |
| 10,000    | 4.94 s       | 11.1 s              | 212 s         | 2.3×            |
| 100,000   | 193 s        | 429 s               | (not run)     | 2.2×            |
| 1,000,000 | 12,737 s     | 18,509 s (10×100k)  | (not run)     | 1.45×           |

Peak RAM at 1M: antiGen ~116 GB (dense L×M buffer), usher-sampled ~4.5 GB.

antiGen-data is fastest at every N tested. Both tools scale ~O(N²) at the top end,
but the GPU lead **narrows** with N (4.8× → 1.45×) as antiGen's brute-force
score-against-every-edge cost catches up — extrapolating, the curves cross not far
beyond 1M. classic usher (strict placement-only analog) is impractical past ~10k.
See `scaling.png`.

Caveat for the 1M usher-sampled point: usher-sampled crashes placing 1M in a single
batch, so it is run as 10 incremental 100k batches (with per-batch SPR refinement) —
more work than antiGen's single-shot placement, so the 1.45× is conservative for
antiGen. See the batching note above.

## File manifest

Tracked: `bench/*.py`, `bench/*.sh`, `results*.csv`, `scaling.png`, this README;
SLURM job scripts live in `../slurm/`.
Git-ignored (large/derived, regenerable): `data/` (downloads, VCFs, FASTAs, per-run
logs/outputs), `guide.{nwk,vcf,pb}`, `usher_guide_out/`, `out_smoke/`, smoke inputs.
