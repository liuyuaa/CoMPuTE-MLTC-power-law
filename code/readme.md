# Power-law scaling of multiple long-term condition accumulation over the life course

This repository contains the analysis scripts used to construct mean cumulative count (MCC) trajectories and fit the scaling models reported in the study. The workflow is organised into four sequential scripts, each corresponding to a distinct analytical step.

---

## Workflow Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_MCC_bootstrapping.py` | Build stratified MCC trajectories and bootstrap replicates |
| 2 | `2_power_law_ols_gls_fitting.py` | Fit the power-law model using OLS and GLS, with either raw-scale or log-log-scale fitting |
| 3 | `3_second_order_fitting.py` | Fit and compare first- and second-order log-polynomial models |
| 4 | `4_exponential_fitting.py` | Compare power-law and exponential models |

Scripts 2–4 all depend on outputs generated in Step 1 and should be run in the order listed above.

---

## Running Environment

**Operating system:** Ubuntu 22.04.5 LTS (other Linux distributions may also work)

**Python version:** 3.8.18

**RAM:** 256 GB (workstation used for development and testing; smaller datasets may require less)

### Python Dependencies

| Package | Tested version |
|---------|----------------|
| numpy | 1.24.3 |
| pandas | 1.5.3 |
| scipy | 1.10.1 |
| statsmodels | 0.14.1 |

---

## Repository Structure

```text
./
├── 1_MCC_bootstrapping.py
├── 2_power_law_ols_gls_fitting.py
├── 3_second_order_fitting.py
├── 4_exponential_fitting.py
├── README.md
└── data/
    └── df_patient_20_80.csv
```

---

## Input Data

The main input file for Step 1 is, by default:

```text
./data/df_patient_20_80.csv
```

### Required columns

**Demographic and follow-up variables**

| Column | Description |
|--------|-------------|
| `patid` | Unique patient identifier |
| `gender` | Biological sex (`M` = Male, `F` = Female) |
| `yob` | Year of birth |
| `exit_age` | Age at end of follow-up |
| `1st_cond` | First recorded chronic condition (index condition) |
| `1st_age` | Age at onset of index condition |

**Condition columns**

The following 18 condition columns must be present:

```python
['Anx', 'Dep', 'SMI', 'Ast', 'COPD', 'Diab', 'Hyp', 'CHD', 'StroTIA', 'AF', 'HF', 'PAD', 'CKD', 'Dem', 'Park', 'Ost', 'RA', 'Can']
```

Each column should contain a diagnosis date (or date-like value) parseable by `pandas.to_datetime`. Age at onset is derived by subtracting `yob` from the diagnosis year.

### Stratification

The cohort is stratified by `gender`, `1st_cond`, and `1st_age_band`. The age bands are:

```python
[(20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79)]
```

Only individuals whose index condition onset falls within one of these bands are retained.

---

## Step 1: Build MCC Trajectories and Bootstrap Replicates

**Script:** `1_MCC_bootstrapping.py`

```bash
python 1_MCC_bootstrapping.py \
  --input ./data/df_patient_filter_all.csv \
  --output ./results_boot/ \
  --id_col patid \
  --index_age_col 1st_age \
  --exit_col exit_age \
  --n_boot 100 \
  --n_jobs 8
```

This script builds MCC trajectories for each stratum, computes 100 bootstrap replicates at the individual level, and saves each stratum as a compressed `.npz` file together with a summary index file for use in downstream fitting steps.

**Output directory: `./results_boot/`**

**Key files:**

- `bootstrap_mcc_index.csv` — one row per stratum, with columns including:

| Column | Description |
|--------|-------------|
| `gender` | Sex of the stratum |
| `1st_cond` | Index condition |
| `1st_age_band` | Onset age band |
| `stratum_name` | Unique stratum identifier |
| `npz_path` | Path to the stratum `.npz` file |
| `n_t` | Number of time points |
| `t_ref` | Landmark age |
| `mcc_ref`, `var_ref` | MCC and its variance at the landmark |
| `n_people` | Number of individuals in the stratum |

- One `.npz` file per stratum (e.g., `M_Hyp_20_29.npz`, `F_Dep_40_49.npz`), each containing:

| Array | Description |
|-------|-------------|
| `t` | Attained age grid |
| `Y` | Risk set size at each age |
| `mcc` | MCC point estimate |
| `mcc_var`, `mcc_se` | Variance and standard error of MCC |
| `t_ref` | Landmark age |
| `mcc_ref`, `var_ref` | Reference MCC and variance at landmark |
| `delta_t` | Years elapsed since the landmark |
| `delta_mcc` | Post-landmark MCC increment |
| `var_delta_mcc`, `se_delta_mcc` | Variance and standard error of `delta_mcc` |
| `mcc_lo`, `mcc_hi` | 95% bootstrap interval for MCC |
| `delta_mcc_lo`, `delta_mcc_hi` | 95% bootstrap interval for `delta_mcc` |
| `boot_mcc` | Bootstrap MCC curves |
| `boot_delta_mcc` | Bootstrap `delta_mcc` curves |

**This step is complete when** `bootstrap_mcc_index.csv` is non-empty and the number of `.npz` files matches the number of rows in that file.

---

## Step 2: Fit the First-Order Power-Law Model

**Script:** `2_power_law_ols_gls_fitting.py`

```bash
python 2_power_law_ols_gls_fitting.py \
  --index_csv ./results_boot/bootstrap_mcc_index.csv \
  --output ./results_fit \
  --fit_space raw \
  --methods all \
  --min_delta_t 1.0
```

This script fits the first-order power-law model $\Delta\text{MCC}(\Delta t) = \alpha \cdot (\Delta t)^{\beta}$ to each stratum using ordinary least squares (OLS), generalised least squares (GLS), or both (`--methods all`). Fitting can be performed on the original scale (`--fit_space raw`) or the log-log scale (`--fit_space log`).

**Output directory: `./results_fit/raw__all/`**

**Key files:**

- `fit_results_long.csv` — one row per stratum–method combination, with columns including `stratum_name`, `method` (`ols` or `gls`), `alpha`, `beta`, `r2` (OLS), `gr2` (GLS), and metadata from `bootstrap_mcc_index.csv`
- `fit_results_side_by_side_with_meta.csv` — one row per stratum with OLS and GLS estimates shown side by side
- `fit_summary_by_method.csv` — method-level summaries including number of fitted strata, median `alpha`, median `beta`, and median `r2`/`gr2`
- `fit_curves/*.npz` — stratum-level curve details, including `x_fit`, `y_obs`, `cov_raw`, and fitted values `yhat_ols` and/or `yhat_gls`

**This step is complete when** the run directory exists, `fit_results_long.csv` is non-empty, and each fitted stratum has a corresponding file in `fit_curves/`.

---

## Step 3: Fit First- and Second-Order Log-Polynomial Models

**Script:** `3_second_order_fitting.py`

```bash
python 3_second_order_fitting.py \
  --index_csv ./results_boot/bootstrap_mcc_index.csv \
  --output ./results_fit_log_poly \
  --orders 1,2 \
  --min_delta_t 1.0 \
  --bootstrap_n_use 100 \
  --random_state 42
```

This script fits and compares two log-polynomial models using GLS:

- **First-order** (equivalent to the primary power-law): $\log \Delta\text{MCC}(\Delta t) = \gamma + \beta_1 \log(\Delta t)$
- **Second-order** (allows additional curvature): $\log \Delta\text{MCC}(\Delta t) = \gamma + \beta_1 \log(\Delta t) + \beta_2 [\log(\Delta t)]^2$

**Output directory: `./results_fit_log_poly/log_poly__gls_order_1-2/`**

**Key files:**

- `fit_results_long.csv` — one row per stratum–order combination, with columns including `stratum_name`, `order` (`1` or `2`), `gamma`, `alpha`, `beta1`, `beta2` (order-2 only), `bic`, `gr2`, and bootstrap uncertainty columns (`alpha_boot_se`, `alpha_boot_ci_low`, `alpha_boot_ci_high`, `beta1_boot_se`, etc.)
- `fit_results_side_by_side_with_meta.csv` — one row per stratum with first- and second-order results side by side
- `fit_summary_by_method_order.csv` — summaries reported separately for order 1 and order 2
- `fit_curves/*.npz` — stratum-level outputs including `x_fit`, `y_obs`, `cov_raw`, `cov_log`, `yhat_gls_order1`, `yhat_gls_order2`, `params_gls_order1`, `params_gls_order2`, and optional bootstrap parameter arrays

**This step is complete when** the run directory exists, `fit_results_long.csv` contains rows for both order 1 and order 2, and `fit_summary_by_method_order.csv` has been generated.

---

## Step 4: Compare Power-Law and Exponential Models

**Script:** `4_exponential_fitting.py`

```bash
python 4_exponential_fitting.py \
  --index_csv ./results_boot/bootstrap_mcc_index.csv \
  --output ./results_fit_power_exp \
  --models powerlaw,exponential \
  --min_delta_t 1.0 \
  --bootstrap_n_use 100 \
  --random_state 42
```

This script fits and compares two GLS models on the original scale:

- **Power law:** $\Delta\text{MCC}(\Delta t) = \alpha \cdot (\Delta t)^{\beta}$
- **Exponential:** $\Delta\text{MCC}(\Delta t) = \alpha \cdot \exp(\beta \Delta t)$

**Output directory: `./results_fit_power_exp/raw_compare__gls_powerlaw-exponential/`**

**Key files:**

- `fit_results_long.csv` — one row per stratum–model combination, with columns including `stratum_name`, `model_name` (`powerlaw` or `exponential`), `alpha`, `beta`, `bic`, `gr2`, and bootstrap uncertainty columns
- `fit_results_side_by_side_with_meta.csv` — one row per stratum with power-law and exponential results side by side
- `fit_summary_by_method_model.csv` — model-level summaries including number of fitted strata, median `alpha`, median `beta`, median `bic`, and median `gr2`
- `fit_curves/*.npz` — stratum-level outputs including `x_fit`, `y_obs`, `cov_raw`, `yhat_gls_powerlaw`, `yhat_gls_exponential`, `params_gls_powerlaw`, `params_gls_exponential`, and optional bootstrap parameter arrays for each model

**This step is complete when** the run directory exists, `fit_results_long.csv` contains rows for both `powerlaw` and `exponential`, and `fit_summary_by_method_model.csv` has been generated.

