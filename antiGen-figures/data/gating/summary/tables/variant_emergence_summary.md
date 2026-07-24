# Variant-emergence precision @ top-decile — trained models, per cutoff

Metric: of the model's top-10% (position, AA) predictions at the BA.2 *variant-root* sequence, the fraction that had **≥100 observed mutations among BA.2 descendants** by the *last* tracked time step (≈6.3 yr since pandemic start). Eval variant = `all`, gap/stop tokens masked (`_nogapstop`), threshold `hi`=100. Only emitted at pre-saturation cutoffs (a BA.2 testing-root example must exist). Bold = row max among trained models. Baselines `pssm_root` / `pssm_root_position` / `EVEscape` and the DMS-ranking references `Omicron_BA{1,2}_{bind,expr}` are in `variant_emergence__all_nogapstop_thrhi__final.csv`.

| cutoff (yr) | ≈date | blosum62 | esmc | transformer | gated_static_esmc_blosum62 | gated_static_esmc_xfmr | gated_uncertainty_esmc_blosum62 | gated_uncertainty_esmc_xfmr | esm2 | gated_static_esm2_xfmr | gated_uncertainty_esm2_xfmr | esm2_35m | gated_static_esm2_35m_xfmr | gated_uncertainty_esm2_35m_xfmr | gated_static_xfmr_pssm | gated_uncertainty_xfmr_pssm | pssm_root | EVEscape |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.352 | 2020-04 | 0.161 | 0.098 | 0.241 | 0.227 | 0.228 | 0.220 | 0.228 | — | — | — | 0.207 | 0.230 | 0.233 | **0.272** | **0.272** | 0.220 | 0.160 |
| 0.681 | 2020-08 | 0.161 | 0.244 | 0.233 | 0.242 | 0.251 | 0.272 | 0.249 | — | — | — | 0.267 | 0.262 | 0.251 | **0.293** | 0.285 | 0.277 | 0.160 |
| 1.010 | 2020-12 | 0.161 | 0.287 | 0.296 | 0.298 | 0.296 | 0.288 | 0.293 | — | — | — | 0.296 | 0.306 | 0.312 | 0.332 | **0.335** | 0.293 | 0.160 |
| 1.338 | 2021-04 | 0.161 | 0.326 | 0.359 | 0.317 | 0.343 | 0.321 | 0.325 | — | — | — | 0.356 | 0.343 | 0.346 | 0.364 | **0.369** | 0.346 | 0.160 |
| 1.667 | 2021-08 | 0.161 | 0.347 | 0.380 | 0.317 | 0.348 | 0.338 | 0.338 | — | — | — | 0.374 | 0.353 | 0.353 | **0.385** | 0.374 | 0.359 | 0.160 |
| 1.995 | 2021-12 | 0.161 | 0.335 | 0.382 | 0.348 | 0.346 | 0.366 | 0.340 | — | — | — | 0.372 | 0.356 | 0.359 | **0.390** | 0.382 | 0.364 | 0.160 |
| 2.324 | 2022-04 | 0.159 | 0.381 | 0.395 | 0.351 | 0.359 | 0.353 | 0.361 | — | — | — | 0.390 | 0.369 | 0.369 | **0.411** | 0.403 | 0.382 | 0.157 |
| 2.653 | 2022-08 | 0.140 | 0.359 | 0.361 | 0.335 | 0.340 | 0.353 | 0.353 | 0.353 | 0.351 | **0.369** | 0.356 | 0.343 | 0.361 | 0.366 | 0.351 | 0.356 | 0.147 |
| 2.981 | 2022-12 | 0.120 | 0.327 | 0.322 | 0.304 | 0.319 | 0.322 | 0.312 | 0.338 | 0.327 | 0.335 | **0.340** | 0.325 | 0.327 | 0.338 | 0.325 | 0.338 | 0.134 |
