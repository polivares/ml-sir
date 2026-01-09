# Benchmark Analysis (from `runs/summary_simplified.csv`)

This document analyzes the results contained in `runs/summary_simplified.csv`, with the goal of comparing
**traditional (optimization / likelihood-based) SIR parameter inference** against **neural-network-based inference**
for estimating `(beta, gamma)` from infected trajectories.

The **primary comparison metric** in this analysis is **R²** (`r2_beta`, `r2_gamma`). Secondary metrics (MAE/RMSE and timing)
are used to contextualize trade-offs.

All numeric claims in the **main results sections** are traceable to rows in `runs/summary_simplified.csv` via the tuple:
`(exp, scenario, method)` plus the reported metric columns.

To address limitations of the simplified CSV (notably: **Exp3 sweep-point identification** and **selectively excluding known-bad runs**),
this document also cross-references `runs/summary.csv` (which includes `run_dir` and per-run `cfg_*` columns such as the Exp3 sweep value `cfg_rho`).

Note: run directories matching `runs/exp4_20260107_125910_*` were execution errors and are excluded. Exp4 runs under
`runs/exp4_20260108_092011_*` are included.

## Scope, columns, and traceability

**Source file**
- `runs/summary_simplified.csv`

**Additional files used to resolve specific limitations**
- `runs/summary.csv` (same aggregated metrics rows + `run_dir` + `cfg_*` columns such as `cfg_rho`, `cfg_noise`, `cfg_train_mode`)
- `runs/<run_dir>/predictions.npz` (used to quantify Exp4 all-`NaN` neural outputs in clean varinit)

**Columns used (directly from the CSV)**
- Accuracy: `r2_beta`, `r2_gamma`, `mae_beta`, `rmse_beta`, `mae_gamma`, `rmse_gamma`
- Timing: `time_p50`, `time_p90`, `train_time_sec`
- Context: `exp`, `exp_md`, `scenario`, `method`, `n_test`

**Derived helper metrics used in this document**
- `r2_mean = (r2_beta + r2_gamma) / 2` (for ranking; computed from CSV columns, not stored in the file).
- `mae_total = mae_beta + mae_gamma` (secondary; computed from CSV columns, not stored in the file).

**How to reproduce any table/claim**
```python
import pandas as pd

# Use runs/summary.csv to filter out known-bad run directories (Exp4 20260107).
df_full = pd.read_csv("runs/summary.csv")
df_full = df_full[~df_full["run_dir"].astype(str).str.startswith("runs/exp4_20260107_125910_")].copy()

# Keep the same columns as runs/summary_simplified.csv for analysis.
cols = [
  "mae_beta","rmse_beta","r2_beta",
  "mae_gamma","rmse_gamma","r2_gamma",
  "time_p50","time_p90",
  "method","scenario","n_test","train_time_sec",
  "exp","exp_md",
]
df = df_full[cols].copy()

df["r2_mean"] = (df["r2_beta"] + df["r2_gamma"]) / 2
df["mae_total"] = df["mae_beta"] + df["mae_gamma"]

# Example: locate Exp1 noisy-test Poisson rows and sort by r2_mean (descending)
(
  df.query("exp == 'exp1' and scenario == 'train_noisy_test_poisson'")
    .sort_values("r2_mean", ascending=False)
    .loc[:, ["method","r2_beta","r2_gamma","r2_mean","time_p50","train_time_sec","mae_beta","mae_gamma"]]
)
```

**How to map a row to its `run_dir` / config (when needed)**
```python
import pandas as pd

df_full = pd.read_csv("runs/summary.csv")
df_full = df_full[~df_full["run_dir"].astype(str).str.startswith("runs/exp4_20260107_125910_")].copy()

# Example: Exp3 row-level sweep mapping (rho is stored as cfg_rho in runs/summary.csv)
(
  df_full.query("exp == 'exp3' and scenario == 'train_noisy_test_poisson' and method == 'mlp_hetero'")
         .loc[:, ["run_dir","cfg_rho","r2_beta","r2_gamma","time_p50","train_time_sec"]]
         .sort_values("cfg_rho")
)
```

## Dataset snapshot

Note: all tables/statistics below exclude only run directories matching `runs/exp4_20260107_125910_*` (see note in the introduction).

**Rows by experiment (`exp`)**
| exp | rows |
| --- | --- |
| exp0 | 96 |
| exp1 | 141 |
| exp2 | 19 |
| exp3 | 120 |
| exp4 | 24 |

**Rows by scenario (`scenario`)**
| scenario | rows |
| --- | --- |
| clean | 96 |
| train_mixed_test_poisson | 55 |
| train_clean_test_poisson | 42 |
| train_noisy_test_poisson | 41 |
| train_clean_test_negbin | 41 |
| train_noisy_test_negbin | 41 |
| train_mixed_test_negbin | 41 |
| window_downsample | 19 |
| varinit_frac_append_fraction_train_clean_test_poisson | 4 |
| varinit_frac_append_fraction_train_noisy_test_poisson | 4 |
| varinit_frac_append_fraction_train_mixed_test_poisson | 4 |
| varinit_frac_append_fraction_train_clean_test_negbin | 4 |
| varinit_frac_append_fraction_train_noisy_test_negbin | 4 |
| varinit_frac_append_fraction_train_mixed_test_negbin | 4 |

**Rows by method (`method`)**
| method | rows |
| --- | --- |
| mlp_hetero | 49 |
| mlp_mdn | 49 |
| baseline_mle_negbin_de | 42 |
| baseline_mle_poisson | 42 |
| mlp_branched | 14 |
| mlp | 14 |
| linear | 13 |
| lstm | 13 |
| gru | 13 |
| attn_cnn | 13 |
| inception | 13 |
| conv_gru | 13 |
| resmlp | 13 |
| cnn1d | 13 |
| tcn | 13 |
| transformer | 13 |
| baseline_huber | 12 |
| baseline_wls | 12 |
| baseline_log_mse | 12 |
| baseline_mse | 6 |
| baseline_mse_de | 6 |
| baseline_mle_negbin | 6 |
| baseline_mle_poisson_de | 6 |

## Method groups used in this analysis

This grouping is based purely on the `method` string in the CSV:

- **Traditional baselines**: methods whose name starts with `baseline_`
- **Neural networks**: all non-`baseline_` methods **excluding** `linear`
- **`linear`**: kept separate as a non-neural regression baseline

## Results (R²-first): traditional vs neural

### 1) Exp0 (clean): clean observations (focus on `n_test=500`)

Filter used: `exp='exp0'`, `scenario='clean'`, `n_test=500`.

**Selected rows (sorted by `r2_mean` descending)**
| method | r2_beta | r2_gamma | r2_mean | mae_beta | mae_gamma | time_p50 | train_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_mse_de | 1 | 1 | 1 | 4.90628e-08 | 2.40199e-08 | 44.8084 |  |
| baseline_mse | 0.999966 | 0.999763 | 0.999865 | 0.00030344 | 0.000266763 | 5.98388 |  |
| baseline_huber | 0.999865 | 0.999081 | 0.999473 | 0.000955107 | 0.000832138 | 7.6482 |  |
| mlp | 0.998535 | 0.992169 | 0.995352 | 0.0201842 | 0.0123637 | 0.0435088 | 24.5195 |
| mlp_branched | 0.998581 | 0.990334 | 0.994458 | 0.0172819 | 0.012878 | 0.0443102 | 22.1564 |
| inception | 0.998688 | 0.989419 | 0.994054 | 0.0162239 | 0.0155215 | 0.061671 | 200.992 |
| resmlp | 0.997866 | 0.988713 | 0.993289 | 0.0193146 | 0.0156114 | 0.04453 | 19.8505 |
| tcn | 0.997 | 0.984931 | 0.990966 | 0.0214652 | 0.016294 | 0.060824 | 242.979 |
| baseline_wls | 0.943011 | 0.969545 | 0.956278 | 0.0252739 | 0.0112121 | 6.61069 |  |
| transformer | 0.87105 | 0.845343 | 0.858197 | 0.0944488 | 0.0526238 | 0.0617366 | 1627.06 |
| baseline_log_mse | 0.65317 | 0.983711 | 0.818441 | 0.0906491 | 0.00613412 | 9.57553 |  |
| linear | 0.395831 | 0.701218 | 0.548525 | 0.504404 | 0.122064 | 0.0434919 | 8.21332 |
| cnn1d | 0.0537917 | 0.368027 | 0.21091 | 0.633136 | 0.180603 | 0.0447687 | 9.44981 |
| lstm | -0.00872326 | -0.00155163 | -0.00513744 | 0.685317 | 0.229963 | 0.0660133 | 264.906 |
| gru | -0.0068152 | -0.00401938 | -0.00541729 | 0.684847 | 0.229919 | 0.0648707 | 185.091 |
| conv_gru | -0.0159639 | -0.00489485 | -0.0104294 | 0.687351 | 0.22992 | 0.0607128 | 25.8211 |

**What the CSV supports (Exp0 clean, `n_test=500`)**
- **R² ceiling**: the best baselines reach near-perfect R² (e.g., `baseline_mse_de` has `r2_beta=1`, `r2_gamma=1`; `baseline_mse` has `r2_mean=0.999865`).
- **Neural competitiveness in R² (but slightly lower)**: the best neural row here is `mlp` with `r2_beta=0.998535`, `r2_gamma=0.992169` (`r2_mean=0.995352`).
- **Architecture sensitivity**: some neural sequence models can fail badly even in this clean setting (e.g., negative `r2_mean` for `gru`, `lstm`, `conv_gru`).
- **Speed gap remains large**: baselines show `time_p50` in seconds-to-tens-of-seconds range, while many neural rows are ~`0.04–0.06` in `time_p50` under the same filter.

### 2) Exp2 (window/downsample): partial observations

Filter used: `exp='exp2'`, `scenario='window_downsample'`, `n_test=500`.

**Selected rows (sorted by `r2_mean` descending)**
| method | r2_beta | r2_gamma | r2_mean | mae_beta | mae_gamma | time_p50 | train_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_mse_de | 1 | 1 | 1 | 9.70174e-08 | 6.22161e-08 | 10.6334 |  |
| baseline_mse | 0.999984 | 0.999888 | 0.999936 | 0.000209129 | 0.000183429 | 1.23629 |  |
| baseline_huber | 0.999923 | 0.999471 | 0.999697 | 0.000503753 | 0.000439406 | 1.72636 |  |
| resmlp | 0.998549 | 0.99339 | 0.995969 | 0.0204061 | 0.0120615 | 0.0443788 | 27.6705 |
| conv_gru | 0.998374 | 0.991855 | 0.995115 | 0.0159711 | 0.0140372 | 0.0502742 | 63.7919 |
| inception | 0.998192 | 0.991556 | 0.994874 | 0.0172372 | 0.0116477 | 0.046776 | 25.0842 |
| tcn | 0.997516 | 0.984872 | 0.991194 | 0.0215087 | 0.0141505 | 0.0454696 | 30.4991 |
| mlp_branched | 0.996627 | 0.985333 | 0.99098 | 0.031843 | 0.0137016 | 0.0438788 | 22.8493 |
| mlp | 0.996406 | 0.979027 | 0.987717 | 0.028877 | 0.0210353 | 0.043917 | 14.3697 |
| transformer | 0.991885 | 0.976291 | 0.984088 | 0.0427384 | 0.0186463 | 0.0492084 | 31.7302 |
| baseline_wls | 0.922485 | 0.984564 | 0.953525 | 0.0245171 | 0.00675654 | 1.45539 |  |
| baseline_log_mse | 0.678951 | 0.984218 | 0.831585 | 0.0846684 | 0.00574065 | 1.96688 |  |
| linear | 0.0832636 | 0.387869 | 0.235566 | 0.64721 | 0.178605 | 0.0432039 | 9.42974 |

**What the CSV supports (Exp2 window/downsample)**
- **R² remains very high for top baselines** (`baseline_mse_de` / `baseline_mse` / `baseline_huber`).
- **Strong neural R² but below the best baselines**: e.g., `resmlp` has `r2_mean=0.995969` while `baseline_mse` has `r2_mean=0.999936` under the same filter.
- **Inference-time advantage persists for neural methods** (sub-second `time_p50`) relative to baselines (`time_p50` ~`1–10` here, depending on baseline).

### 3) Exp1 (noise): which side wins on R² depends on the train/test noise regime

All Exp1 rows in this file have `n_test=500`. For each Exp1 `scenario`, we compare:
- the best baseline row (maximum `r2_mean` among `baseline_*`), and
- the best neural row (maximum `r2_mean` among non-`baseline_`, excluding `linear`).

| scenario | best_baseline | b_r2_beta | b_r2_gamma | b_r2_mean | b_time_p50_s | best_nn | nn_r2_beta | nn_r2_gamma | nn_r2_mean | nn_time_p50_s | nn_train_s | delta_r2_mean(nn-b) | speedup_p50(b/nn) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_clean_test_negbin | baseline_mle_negbin_de | 0.966599 | 0.905417 | 0.936008 | 7.1704 | mlp_mdn | 0.844481 | -0.0917249 | 0.376378 | 0.0661931 | 37.108 | -0.55963 | 108.326 |
| train_clean_test_poisson | baseline_mle_poisson_de | 0.973 | 0.897861 | 0.93543 | 7.30981 | mlp_mdn | 0.925602 | 0.695198 | 0.8104 | 0.0617907 | 32.113 | -0.12503 | 118.3 |
| train_mixed_test_negbin | baseline_mle_negbin_de | 0.958836 | 0.888872 | 0.923854 | 7.22871 | mlp_hetero | 0.934241 | 0.894278 | 0.91426 | 0.0590578 | 20.1553 | -0.00959424 | 122.401 |
| train_mixed_test_poisson | baseline_mle_negbin_de | 0.978371 | 0.931556 | 0.954963 | 7.56638 | mlp_mdn | 0.961053 | 0.921462 | 0.941258 | 0.0591485 | 22.0837 | -0.0137057 | 127.922 |
| train_noisy_test_negbin | baseline_mle_negbin_de | 0.966928 | 0.913086 | 0.940007 | 8.67403 | tcn | 0.9513 | 0.929034 | 0.940167 | 0.0570469 | 187.512 | 0.000159808 | 152.051 |
| train_noisy_test_poisson | baseline_mle_negbin_de | 0.970069 | 0.897139 | 0.933604 | 7.52105 | transformer | 0.958186 | 0.938928 | 0.948557 | 0.0642143 | 2739.78 | 0.0149535 | 117.124 |

**What the CSV supports (Exp1)**
- **Neural wins on R² in some noisy-test regimes**:
  - `exp1` + `train_noisy_test_poisson`: `transformer` has `r2_mean=0.948557`, higher than the best baseline (`baseline_mle_negbin_de` with `r2_mean=0.933604`).
  - `exp1` + `train_noisy_test_negbin`: `tcn` has `r2_mean=0.940167`, slightly higher than the best baseline (`baseline_mle_negbin_de` with `r2_mean=0.940007`).
- **Baselines dominate when training is clean but test is noisy**:
  - `exp1` + `train_clean_test_negbin`: best baseline has `r2_mean=0.936008`, while the best neural row (`mlp_mdn`) has `r2_mean=0.376378` driven by `r2_gamma=-0.0917249`.
- **Training-time trade-off is visible in the same scenario**:
  - `exp1` + `train_noisy_test_poisson`: the top R² method `transformer` has `train_time_sec=2739.78` (in the CSV) while faster-to-train neural methods exist in the same filter (e.g., `tcn` in that scenario has `r2_mean=0.948412` and `train_time_sec=182.375`, both values present in the CSV).
- **Inference-time advantage is consistent**: neural `time_p50` values are ~`0.06`, while the best-R² baselines are ~`7–9` in `time_p50` in these scenarios.

**Notable neural failure example (Exp1)**
- `exp='exp1'`, `scenario='train_clean_test_negbin'`, `method='attn_cnn'`: `r2_beta=-25.511280`, `r2_gamma=-17.823050` (both in the CSV), indicating catastrophic generalization for that architecture under that condition.

### 4) Exp3 (rho sweep): robustness across sweep points

In Exp3, each `(scenario, method)` has `n=5` rows (different sweep points). `runs/summary_simplified.csv` does not expose which row
corresponds to which sweep value (the sweep parameter is `rho`), so the table below summarizes variability using **min/median/max**
of `r2_mean` across those 5 rows.
To remain traceable, we summarize variability using **min/median/max** of `r2_mean` across those 5 rows, plus the worst-case
`r2_beta` / `r2_gamma` within the group.

| scenario | method | n | r2_beta_min | r2_gamma_min | r2_mean_min | r2_mean_median | r2_mean_max | time_p50_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_clean_test_negbin | baseline_mle_negbin_de | 5 | 0.925344 | 0.841645 | 0.883495 | 0.936008 | 0.951129 | 7.32811 |
| train_clean_test_negbin | baseline_mle_poisson | 5 | 0.925083 | 0.842452 | 0.883767 | 0.933227 | 0.949804 | 5.82607 |
| train_clean_test_negbin | mlp_hetero | 5 | -1.28599 | -0.0479835 | -0.666984 | 0.536749 | 0.896291 | 0.0436561 |
| train_clean_test_negbin | mlp_mdn | 5 | -0.403941 | -0.683965 | -0.543953 | 0.752627 | 0.860526 | 0.0437751 |
| train_clean_test_poisson | baseline_mle_negbin_de | 5 | 0.933689 | 0.849784 | 0.891737 | 0.93532 | 0.95169 | 7.78734 |
| train_clean_test_poisson | baseline_mle_poisson | 5 | 0.934068 | 0.848652 | 0.89136 | 0.935429 | 0.953535 | 5.98263 |
| train_clean_test_poisson | mlp_hetero | 5 | -1.32832 | -0.0464375 | -0.687381 | 0.508144 | 0.926643 | 0.0450485 |
| train_clean_test_poisson | mlp_mdn | 5 | -0.431917 | -0.746028 | -0.588973 | 0.737454 | 0.923671 | 0.0451706 |
| train_mixed_test_negbin | baseline_mle_negbin_de | 5 | 0.917461 | 0.845626 | 0.881543 | 0.931681 | 0.952525 | 7.74988 |
| train_mixed_test_negbin | baseline_mle_poisson | 5 | 0.916123 | 0.84422 | 0.880172 | 0.927579 | 0.954494 | 6.0755 |
| train_mixed_test_negbin | mlp_hetero | 5 | 0.0943407 | 0.475116 | 0.284728 | 0.915968 | 0.931768 | 0.044725 |
| train_mixed_test_negbin | mlp_mdn | 5 | 0.0168137 | 0.476155 | 0.246484 | 0.905747 | 0.930407 | 0.0442067 |
| train_mixed_test_poisson | baseline_mle_negbin_de | 5 | 0.934052 | 0.838995 | 0.886523 | 0.945462 | 0.954963 | 7.53782 |
| train_mixed_test_poisson | baseline_mle_poisson | 5 | 0.934418 | 0.838807 | 0.886612 | 0.943709 | 0.954855 | 5.94096 |
| train_mixed_test_poisson | mlp_hetero | 5 | 0.183107 | 0.452726 | 0.317917 | 0.935262 | 0.94661 | 0.0455547 |
| train_mixed_test_poisson | mlp_mdn | 5 | 0.136455 | 0.371427 | 0.253941 | 0.925867 | 0.941441 | 0.045314 |
| train_noisy_test_negbin | baseline_mle_negbin_de | 5 | 0.937388 | 0.869408 | 0.903398 | 0.931817 | 0.940574 | 7.29309 |
| train_noisy_test_negbin | baseline_mle_poisson | 5 | 0.936698 | 0.868352 | 0.902525 | 0.930764 | 0.939327 | 5.85188 |
| train_noisy_test_negbin | mlp_hetero | 5 | 0.550447 | -6.21717 | -2.83336 | 0.926191 | 0.944424 | 0.0437575 |
| train_noisy_test_negbin | mlp_mdn | 5 | 0.930778 | 0.887772 | 0.913102 | 0.931108 | 0.942049 | 0.0438499 |
| train_noisy_test_poisson | baseline_mle_negbin_de | 5 | 0.93744 | 0.842487 | 0.889963 | 0.944835 | 0.95635 | 7.60287 |
| train_noisy_test_poisson | baseline_mle_poisson | 5 | 0.936685 | 0.841672 | 0.889178 | 0.944857 | 0.953358 | 5.87324 |
| train_noisy_test_poisson | mlp_hetero | 5 | -1.77379 | -271.642 | -136.708 | 0.938293 | 0.95346 | 0.0451461 |
| train_noisy_test_poisson | mlp_mdn | 5 | 0.922195 | 0.895758 | 0.908977 | 0.944824 | 0.955491 | 0.0453146 |

**What the CSV supports (Exp3 robustness)**
- **Median R² can be close in noisy scenarios**:
  - `exp3` + `train_noisy_test_poisson`: `baseline_mle_negbin_de` has `r2_mean_median=0.944835` and `mlp_mdn` has `r2_mean_median=0.944824` (almost identical).
- **But neural tail risk can be extreme**:
  - `exp3` + `train_noisy_test_poisson`: `mlp_hetero` has `r2_mean_min=-136.708` with `r2_gamma_min=-271.642` (catastrophic failure on at least one sweep point).
  - `exp3` + `train_noisy_test_negbin`: `mlp_hetero` has `r2_mean_min=-2.83336` driven by `r2_gamma_min=-6.21717`.
- **Speed remains strongly in favor of neural methods** (`time_p50_median` ~`0.044–0.045`) compared to baselines (~`5.8–7.8`).

**Supplement (Exp3): mapping sweep points to `rho`**

`runs/summary.csv` includes the sweep value (`cfg_rho`) and `run_dir`, so each Exp3 row can be mapped to its exact `rho`.
Below is `r2_mean` for each `(scenario, method, rho)` (computed as `(r2_beta + r2_gamma)/2` from `runs/summary.csv`):

| scenario | method | rho=0.1 | rho=0.3 | rho=0.5 | rho=0.7 | rho=1 |
| --- | --- | --- | --- | --- | --- | --- |
| train_clean_test_negbin | baseline_mle_negbin_de | 0.883495 | 0.930318 | 0.936008 | 0.942543 | 0.951129 |
| train_clean_test_negbin | baseline_mle_poisson | 0.883767 | 0.926508 | 0.933227 | 0.941802 | 0.949804 |
| train_clean_test_negbin | mlp_hetero | -0.666984 | 0.090424 | 0.536749 | 0.793910 | 0.896291 |
| train_clean_test_negbin | mlp_mdn | -0.543953 | 0.429561 | 0.752627 | 0.860526 | 0.854828 |
| train_clean_test_poisson | baseline_mle_negbin_de | 0.891737 | 0.924929 | 0.935320 | 0.951690 | 0.942576 |
| train_clean_test_poisson | baseline_mle_poisson | 0.891360 | 0.923321 | 0.935429 | 0.953535 | 0.943131 |
| train_clean_test_poisson | mlp_hetero | -0.687381 | 0.060604 | 0.508144 | 0.761119 | 0.926643 |
| train_clean_test_poisson | mlp_mdn | -0.588973 | 0.417034 | 0.737454 | 0.860952 | 0.923671 |
| train_mixed_test_negbin | baseline_mle_negbin_de | 0.881543 | 0.918896 | 0.938590 | 0.931681 | 0.952525 |
| train_mixed_test_negbin | baseline_mle_poisson | 0.880172 | 0.917729 | 0.938818 | 0.927579 | 0.954494 |
| train_mixed_test_negbin | mlp_hetero | 0.284728 | 0.887749 | 0.915968 | 0.931768 | 0.918887 |
| train_mixed_test_negbin | mlp_mdn | 0.246484 | 0.861427 | 0.910829 | 0.930407 | 0.905747 |
| train_mixed_test_poisson | baseline_mle_negbin_de | 0.886523 | 0.926417 | 0.954963 | 0.945462 | 0.951641 |
| train_mixed_test_poisson | baseline_mle_poisson | 0.886612 | 0.925371 | 0.954855 | 0.943709 | 0.949542 |
| train_mixed_test_poisson | mlp_hetero | 0.317917 | 0.900003 | 0.935262 | 0.939447 | 0.946610 |
| train_mixed_test_poisson | mlp_mdn | 0.253941 | 0.879642 | 0.925867 | 0.935502 | 0.941441 |
| train_noisy_test_negbin | baseline_mle_negbin_de | 0.903398 | 0.931817 | 0.940007 | 0.930744 | 0.940574 |
| train_noisy_test_negbin | baseline_mle_poisson | 0.902525 | 0.930764 | 0.939327 | 0.927363 | 0.936282 |
| train_noisy_test_negbin | mlp_hetero | 0.922730 | 0.926191 | 0.944424 | 0.932429 | -2.833362 |
| train_noisy_test_negbin | mlp_mdn | 0.925974 | 0.937367 | 0.931108 | 0.913102 | 0.942049 |
| train_noisy_test_poisson | baseline_mle_negbin_de | 0.889963 | 0.944835 | 0.933604 | 0.946803 | 0.956350 |
| train_noisy_test_poisson | baseline_mle_poisson | 0.889178 | 0.944857 | 0.932863 | 0.947483 | 0.953358 |
| train_noisy_test_poisson | mlp_hetero | 0.922593 | 0.938293 | 0.952118 | 0.953460 | -136.708136 |
| train_noisy_test_poisson | mlp_mdn | 0.924895 | 0.944824 | 0.955491 | 0.948231 | 0.908977 |

**Exp3 outlier rows (from `runs/summary.csv`)**

These two rows are the source of the extreme tail failures reported above (use `run_dir` to locate the exact run artifacts):

| exp | scenario | method | cfg_rho | run_dir | r2_beta | r2_gamma | r2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp3 | train_noisy_test_poisson | mlp_hetero | 1.0 | runs/exp3_20260104_090550_06_poisson_noisy_rho1 | -1.773786 | -271.642487 | -136.708136 |
| exp3 | train_noisy_test_negbin | mlp_hetero | 1.0 | runs/exp3_20260104_090550_21_negbin_noisy_rho1_k10 | 0.550447 | -6.217172 | -2.833362 |

### 5) Exp4 (variable init): baselines collapse; neural is “less bad” in noisy/mixed; NaNs in clean

For Exp4 we **exclude** run directories matching `runs/exp4_20260107_125910_*` (execution errors) and **include** the Exp4 runs under
`runs/exp4_20260108_092011_*`.

Since Exp4 has `n=1` per `(scenario, method)` in this filtered set, we report the raw rows (with `r2_mean = (r2_beta + r2_gamma)/2`):

| run_dir | scenario | method | r2_beta | r2_gamma | r2_mean | mae_beta | mae_gamma | time_p50 | train_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `runs/exp4_20260108_092011_04_negbin_clean_rho1_k10` | varinit_frac_append_fraction_train_clean_test_negbin | baseline_mle_negbin_de | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 17.8282 |  |
| `runs/exp4_20260108_092011_04_negbin_clean_rho1_k10` | varinit_frac_append_fraction_train_clean_test_negbin | baseline_mle_poisson | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 0.819631 |  |
| `runs/exp4_20260108_092011_04_negbin_clean_rho1_k10` | varinit_frac_append_fraction_train_clean_test_negbin | mlp_hetero |  |  |  |  |  | 0.0364137 | 4.38589 |
| `runs/exp4_20260108_092011_04_negbin_clean_rho1_k10` | varinit_frac_append_fraction_train_clean_test_negbin | mlp_mdn |  |  |  |  |  | 0.0364632 | 4.46991 |
| `runs/exp4_20260108_092011_01_poisson_clean_rho1` | varinit_frac_append_fraction_train_clean_test_poisson | baseline_mle_negbin_de | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 20.0974 |  |
| `runs/exp4_20260108_092011_01_poisson_clean_rho1` | varinit_frac_append_fraction_train_clean_test_poisson | baseline_mle_poisson | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 1.12513 |  |
| `runs/exp4_20260108_092011_01_poisson_clean_rho1` | varinit_frac_append_fraction_train_clean_test_poisson | mlp_hetero |  |  |  |  |  | 0.0528413 | 7.60124 |
| `runs/exp4_20260108_092011_01_poisson_clean_rho1` | varinit_frac_append_fraction_train_clean_test_poisson | mlp_mdn |  |  |  |  |  | 0.0544402 | 7.74802 |
| `runs/exp4_20260108_092011_06_negbin_mixed_rho1_k10` | varinit_frac_append_fraction_train_mixed_test_negbin | baseline_mle_negbin_de | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 18.1055 |  |
| `runs/exp4_20260108_092011_06_negbin_mixed_rho1_k10` | varinit_frac_append_fraction_train_mixed_test_negbin | baseline_mle_poisson | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 0.827593 |  |
| `runs/exp4_20260108_092011_06_negbin_mixed_rho1_k10` | varinit_frac_append_fraction_train_mixed_test_negbin | mlp_hetero | 0.000274599 | -0.000855446 | -0.000290424 | 0.714554 | 0.22561 | 0.0370397 | 4.97026 |
| `runs/exp4_20260108_092011_06_negbin_mixed_rho1_k10` | varinit_frac_append_fraction_train_mixed_test_negbin | mlp_mdn | -0.00144303 | -0.0031867 | -0.00231487 | 0.714934 | 0.226157 | 0.03609 | 17.7538 |
| `runs/exp4_20260108_092011_03_poisson_mixed_rho1` | varinit_frac_append_fraction_train_mixed_test_poisson | baseline_mle_negbin_de | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 17.6049 |  |
| `runs/exp4_20260108_092011_03_poisson_mixed_rho1` | varinit_frac_append_fraction_train_mixed_test_poisson | baseline_mle_poisson | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 0.833798 |  |
| `runs/exp4_20260108_092011_03_poisson_mixed_rho1` | varinit_frac_append_fraction_train_mixed_test_poisson | mlp_hetero | 0.000274599 | -0.000855446 | -0.000290424 | 0.714554 | 0.22561 | 0.0366132 | 4.96636 |
| `runs/exp4_20260108_092011_03_poisson_mixed_rho1` | varinit_frac_append_fraction_train_mixed_test_poisson | mlp_mdn | -0.00144303 | -0.0031867 | -0.00231487 | 0.714934 | 0.226157 | 0.0363998 | 17.7146 |
| `runs/exp4_20260108_092011_05_negbin_noisy_rho1_k10` | varinit_frac_append_fraction_train_noisy_test_negbin | baseline_mle_negbin_de | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 17.5382 |  |
| `runs/exp4_20260108_092011_05_negbin_noisy_rho1_k10` | varinit_frac_append_fraction_train_noisy_test_negbin | baseline_mle_poisson | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 0.806042 |  |
| `runs/exp4_20260108_092011_05_negbin_noisy_rho1_k10` | varinit_frac_append_fraction_train_noisy_test_negbin | mlp_hetero | 0.000274599 | -0.000855446 | -0.000290424 | 0.714554 | 0.22561 | 0.0365823 | 4.99619 |
| `runs/exp4_20260108_092011_05_negbin_noisy_rho1_k10` | varinit_frac_append_fraction_train_noisy_test_negbin | mlp_mdn | -0.00144303 | -0.0031867 | -0.00231487 | 0.714934 | 0.226157 | 0.036276 | 17.8193 |
| `runs/exp4_20260108_092011_02_poisson_noisy_rho1` | varinit_frac_append_fraction_train_noisy_test_poisson | baseline_mle_negbin_de | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 19.307 |  |
| `runs/exp4_20260108_092011_02_poisson_noisy_rho1` | varinit_frac_append_fraction_train_noisy_test_poisson | baseline_mle_poisson | -2.97889 | -2.82995 | -2.90442 | 1.4273 | 0.43543 | 0.903256 |  |
| `runs/exp4_20260108_092011_02_poisson_noisy_rho1` | varinit_frac_append_fraction_train_noisy_test_poisson | mlp_hetero | 0.000274599 | -0.000855446 | -0.000290424 | 0.714554 | 0.22561 | 0.0382769 | 5.23558 |
| `runs/exp4_20260108_092011_02_poisson_noisy_rho1` | varinit_frac_append_fraction_train_noisy_test_poisson | mlp_mdn | -0.00144303 | -0.0031867 | -0.00231487 | 0.714934 | 0.226157 | 0.038435 | 18.4718 |

**What the data supports (Exp4, filtered)**
- **Baselines are extremely poor by R²**: both `baseline_mle_negbin_de` and `baseline_mle_poisson` have `r2_mean=-2.90442` in every Exp4 scenario shown above.
- **Neural methods are “less bad” in noisy/mixed varinit**: e.g., in `varinit_*_train_mixed_test_negbin`, `mlp_hetero` has `r2_mean=-0.000290424` (near zero) vs baselines at `-2.90442`.
- **Inference-time spread within baselines is large**: in the same Exp4 runs, `baseline_mle_poisson` has `time_p50≈0.81–1.13`, while `baseline_mle_negbin_de` is `time_p50≈17.5–20.1`.

**Supplement (Exp4): clean varinit NaN predictions for neural methods**

In the two clean-varinit runs, the saved `predictions.npz` contains all-`NaN` predictions for both neural methods (`y_pred_mlp_hetero` and `y_pred_mlp_mdn`),
which explains the missing (`NaN`) accuracy metrics in the table above:

| run_dir | scenario | y_pred_mlp_hetero_nan_frac | y_pred_mlp_mdn_nan_frac |
| --- | --- | --- | --- |
| `runs/exp4_20260108_092011_01_poisson_clean_rho1` | varinit_frac_append_fraction_train_clean_test_poisson | 1.0 | 1.0 |
| `runs/exp4_20260108_092011_02_poisson_noisy_rho1` | varinit_frac_append_fraction_train_noisy_test_poisson | 0.0 | 0.0 |
| `runs/exp4_20260108_092011_03_poisson_mixed_rho1` | varinit_frac_append_fraction_train_mixed_test_poisson | 0.0 | 0.0 |
| `runs/exp4_20260108_092011_04_negbin_clean_rho1_k10` | varinit_frac_append_fraction_train_clean_test_negbin | 1.0 | 1.0 |
| `runs/exp4_20260108_092011_05_negbin_noisy_rho1_k10` | varinit_frac_append_fraction_train_noisy_test_negbin | 0.0 | 0.0 |
| `runs/exp4_20260108_092011_06_negbin_mixed_rho1_k10` | varinit_frac_append_fraction_train_mixed_test_negbin | 0.0 | 0.0 |

## Speed and training-cost context (excluding bad Exp4 runs)

### Training time (`train_time_sec`) distribution by method (neural + linear only)

| method | n | min | median | max |
| --- | --- | --- | --- | --- |
| attn_cnn | 13 | 10.9099 | 79.3314 | 170.548 |
| cnn1d | 13 | 9.44981 | 13.3339 | 68.1844 |
| conv_gru | 13 | 6.83571 | 25.4147 | 90.2743 |
| gru | 13 | 7.27831 | 163.708 | 274.794 |
| inception | 13 | 13.3695 | 107.896 | 262.641 |
| linear | 13 | 7.3562 | 9.04888 | 9.85724 |
| lstm | 13 | 7.29455 | 221.742 | 267.649 |
| mlp | 14 | 8.39811 | 14.1592 | 26.7871 |
| mlp_branched | 14 | 8.58035 | 14.4571 | 24.0948 |
| mlp_hetero | 49 | 4.38589 | 14.4963 | 28.4183 |
| mlp_mdn | 49 | 4.46991 | 13.7026 | 37.108 |
| resmlp | 13 | 12.5463 | 14.3573 | 39.2917 |
| tcn | 13 | 19.0623 | 121.796 | 371.444 |
| transformer | 13 | 31.7302 | 2187.68 | 5264.59 |

### Inference time (`time_p50`) distribution by method (all methods)

| method | n | min | median | max |
| --- | --- | --- | --- | --- |
| linear | 13 | 0.0431083 | 0.0440481 | 0.0516826 |
| resmlp | 13 | 0.0436959 | 0.0449244 | 0.0544159 |
| mlp | 14 | 0.0431712 | 0.0451043 | 0.0529915 |
| mlp_branched | 14 | 0.0434898 | 0.0451783 | 0.0567029 |
| mlp_hetero | 49 | 0.0364137 | 0.0451461 | 0.0894772 |
| mlp_mdn | 49 | 0.03609 | 0.0452727 | 0.0833672 |
| cnn1d | 13 | 0.0437043 | 0.0477874 | 0.0795848 |
| tcn | 13 | 0.0444553 | 0.0565513 | 0.0950011 |
| attn_cnn | 13 | 0.0454085 | 0.0605647 | 0.0967495 |
| conv_gru | 13 | 0.046861 | 0.0605779 | 0.0745013 |
| inception | 13 | 0.0447234 | 0.0616324 | 0.111186 |
| transformer | 13 | 0.0490193 | 0.0629566 | 0.0933081 |
| gru | 13 | 0.0491605 | 0.0648707 | 0.113612 |
| lstm | 13 | 0.049676 | 0.0658375 | 0.0729323 |
| baseline_wls | 12 | 1.45539 | 3.45712 | 6.8045 |
| baseline_log_mse | 12 | 1.96688 | 4.66678 | 10.7442 |
| baseline_mle_poisson | 42 | 0.806042 | 5.85674 | 6.73533 |
| baseline_mse | 6 | 1.23629 | 5.86489 | 5.98388 |
| baseline_mle_negbin | 6 | 5.96027 | 6.2114 | 7.4182 |
| baseline_huber | 12 | 1.72636 | 6.79062 | 8.76019 |
| baseline_mle_poisson_de | 6 | 7.04091 | 7.2073 | 8.22536 |
| baseline_mle_negbin_de | 42 | 6.88151 | 7.59163 | 20.0974 |
| baseline_mse_de | 6 | 10.6334 | 46.2204 | 54.3767 |

## Answering the benchmark question (R² as primary metric): should we use neural networks?

Based on the filtered subset of the aggregated results (excluding `runs/exp4_20260107_125910_*`):

**Where baselines are clearly better in R²**
- **Clean / partial-observation regimes** (`exp0` clean, `exp2` window/downsample): the strongest baselines reach `r2_mean≈1`,
  while the best neural rows are slightly lower (e.g., `exp0` clean `mlp` has `r2_mean=0.995352`; `exp2` window/downsample `resmlp` has `r2_mean=0.995969`).
- **Train-clean, test-noisy in Exp1/Exp3**: neural methods can collapse in `r2_gamma` (see `exp1` + `train_clean_test_negbin`, where the best neural row has `r2_gamma=-0.0917249`).

**Where neural networks can match or beat baselines in R² (in this file)**
- **Noisy-test regimes**: `exp1` shows cases where a neural method has higher `r2_mean` than the best baseline in the same scenario:
  - `train_noisy_test_poisson`: `transformer` (`r2_mean=0.948557`) vs best baseline `baseline_mle_negbin_de` (`r2_mean=0.933604`).
  - `train_noisy_test_negbin`: `tcn` (`r2_mean=0.940167`) vs best baseline `baseline_mle_negbin_de` (`r2_mean=0.940007`) — a very small edge.
- **Varinit noisy/mixed** (Exp4): `mlp_hetero` reaches `r2_mean=-0.000290424` (near zero) while baselines are at `r2_mean=-2.90442` (e.g., `runs/exp4_20260108_092011_06_negbin_mixed_rho1_k10`).

**What “good idea” means in practice given these results**
- If the goal is **maximum R² with minimal risk**, the best baselines dominate many scenarios and show fewer catastrophic outliers.
- If the goal is **high-throughput inference** (many curves): neural methods consistently have much smaller `time_p50` (often ~`0.04–0.06`),
  and in some noisy-test regimes they can even improve R².
- However, Exp3 highlights a key downside: some neural methods can have **extreme tail failures** (very negative R²) at some sweep points.
  That risk must be considered if deploying neural inference broadly.

## Limitations visible from the CSV itself

- `runs/summary_simplified.csv` intentionally omits `run_dir` and per-run config columns (`cfg_*`). When row-level mapping matters (e.g., Exp3 sweep points or excluding specific run directories),
  this document uses `runs/summary.csv` and cites `run_dir` explicitly.
- Exp4 clean varinit has **neural prediction failures**: in `runs/exp4_20260108_092011_01_poisson_clean_rho1` and `runs/exp4_20260108_092011_04_negbin_clean_rho1_k10`,
  both `mlp_hetero` and `mlp_mdn` produce all-`NaN` predictions (`nan_frac=1.0` in `predictions.npz`), so R²/MAE/RMSE cannot be computed for those rows.

## Publishability and recommended publication focus

### Are these results publishable “as-is”?

Mostly yes, with two required clarifications in the paper:
1) **Excluded runs**: runs matching `runs/exp4_20260107_125910_*` are excluded as execution errors.
2) **Exp4 clean varinit failures**: neural rows in the two clean-varinit runs have missing accuracy because predictions are all-`NaN` (see Exp4 supplement table).

### What is already strong enough to motivate a paper (Exp0–Exp4)?

The current aggregated results already support a clear, quantitative story about when neural amortized inference is useful and when it is risky:

- **In clean / partial-observation regimes, traditional baselines dominate R²**:
  - Exp0 clean (`n_test=500`): `baseline_mse_de` reaches `r2_mean=1`, while the best neural row shown (`mlp`) has `r2_mean=0.995352`.
  - Exp2 window/downsample: `baseline_mse` has `r2_mean=0.999936` while `resmlp` has `r2_mean=0.995969`.

- **In some noisy-test regimes, neural methods can match or beat the best baseline R² while being much faster at inference**:
  - Exp1 `train_noisy_test_poisson`: `transformer` has `r2_mean=0.948557` vs best baseline `baseline_mle_negbin_de` at `r2_mean=0.933604`,
    with `time_p50` ~`0.064` vs ~`7.52` in the same scenario table.

- **Neural tail-risk is real and can be catastrophic even when median performance is competitive**:
  - Exp3 `train_noisy_test_poisson` at `cfg_rho=1.0`: `mlp_hetero` collapses to `r2_mean=-136.708136` with `r2_gamma=-271.642487`
    in `runs/exp3_20260104_090550_06_poisson_noisy_rho1` (Exp3 outlier table), while the baseline in that same condition remains high
    (`baseline_mle_negbin_de` has `r2_mean=0.956350` in the Exp3 `r2_mean by rho` table).

- **Varinit highlights a different failure mode and a different trade-off**:
  - In Exp4 noisy/mixed varinit, baselines have `r2_mean=-2.90442` while `mlp_hetero` is near zero (`r2_mean=-0.000290424`, e.g., `runs/exp4_20260108_092011_06_negbin_mixed_rho1_k10`).
  - In Exp4 clean varinit, both `mlp_hetero` and `mlp_mdn` output all-`NaN` predictions (`nan_frac=1.0` in `predictions.npz`), so the benchmark must count these as failures unless fixed.

### Recommended focus for a publication

If the goal is a publishable benchmark paper, the strongest angle supported by these results is:

1) **Amortized neural inference vs per-curve optimization**, framed as a trade-off between:
   - **R² (accuracy)** under multiple observation regimes (clean, Poisson/NegBin noise, partial observation via `rho`, window/downsample, varinit), and
   - **inference-time throughput** (`time_p50`), where neural methods are consistently orders of magnitude faster than baselines.

2) **Robustness and failure modes**, not just average performance:
   - Report per-scenario distributions across sweep points (Exp3) and explicitly track outliers with `run_dir` (as done here for `mlp_hetero` at `cfg_rho=1.0`).
   - Treat “invalid prediction” modes (e.g., Exp4 clean-varinit all-`NaN` neural outputs) as first-class benchmark outcomes, because they directly affect deployability.

3) **Where to be careful about conclusions**:
   - In regimes where baselines already achieve near-perfect R² (Exp0/Exp2), the benchmark should emphasize why/when neural speed might still matter,
     rather than claiming accuracy improvements.
   - In regimes with distribution shift (e.g., train-clean/test-noisy), the benchmark should emphasize failure cases as well as “best model” performance
     (see Exp1 `train_clean_test_negbin`, where the best neural row has `r2_gamma=-0.0917249` while the best baseline has `r2_mean=0.936008`).
   - In regimes where R² is near zero (Exp4 noisy/mixed neural rows), include MAE/RMSE alongside R² to avoid over-claiming based on tiny R² differences.
