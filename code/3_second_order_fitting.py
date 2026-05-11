#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

ORDERS_ALL = [1, 2]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_orders(s):
    s = s.strip().lower()
    if s == "all":
        return ORDERS_ALL
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    for v in vals:
        if v not in ORDERS_ALL:
            raise ValueError(f"Unknown order: {v}")
    return vals


def nearest_psd(mat, jitter=1e-10):
    mat = np.asarray(mat, dtype=float)
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.maximum(vals, jitter)
    out = vecs @ np.diag(vals) @ vecs.T
    return 0.5 * (out + out.T)


def pick_bootstrap_n_use(n_total, n_use):
    if n_use is None or n_use <= 0:
        return n_total
    return min(n_total, n_use)


def build_fit_mask(delta_t, delta_y, min_delta_t=1.0):
    delta_t = np.asarray(delta_t, dtype=float)
    delta_y = np.asarray(delta_y, dtype=float)
    mask = np.isfinite(delta_t) & np.isfinite(delta_y)
    mask &= delta_t >= min_delta_t
    mask &= delta_y > 0
    return mask


def make_nested_cov_from_var(var_vec):
    var_vec = np.asarray(var_vec, dtype=float)
    n = len(var_vec)
    cov = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            cov[i, j] = var_vec[min(i, j)]
    return cov


def get_formula_var_cov(npz_obj, mask, fit_space="raw"):
    y_raw = np.asarray(npz_obj["delta_mcc"], dtype=float)[mask]
    var_raw = np.asarray(npz_obj["var_delta_mcc"], dtype=float)[mask]
    var_raw = np.maximum(var_raw, 1e-12)

    cov_raw = make_nested_cov_from_var(var_raw)
    cov_raw = nearest_psd(cov_raw)

    if fit_space == "raw":
        return var_raw, cov_raw

    denom = np.outer(np.maximum(y_raw, 1e-12), np.maximum(y_raw, 1e-12))
    cov_log = cov_raw / denom
    cov_log = nearest_psd(cov_log)
    var_log = np.maximum(np.diag(cov_log), 1e-12)
    return var_log, cov_log


def robust_cholesky(cov):
    cov = nearest_psd(cov)
    try:
        cf = cho_factor(cov, lower=True, check_finite=False)
        return cf, cov
    except LinAlgError:
        cov2 = cov + np.eye(cov.shape[0]) * 1e-8
        cf = cho_factor(cov2, lower=True, check_finite=False)
        return cf, cov2


def generalized_ss(resid, cov):
    cf, cov_used = robust_cholesky(cov)
    val = float(resid.T @ cho_solve(cf, resid, check_finite=False))
    return val, cov_used, cf


def build_design_matrix_log_poly(x, order):
    x = np.asarray(x, dtype=float)
    lx = np.log(x)
    cols = [np.ones_like(lx)]
    for k in range(1, order + 1):
        cols.append(lx ** k)
    return np.column_stack(cols)


def metrics_gls(y_true, y_pred, cov_raw):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred

    grss, _, cf = generalized_ss(resid, cov_raw)

    n = len(y_true)
    ones = np.ones(n, dtype=float)
    sigma_inv_y = cho_solve(cf, y_true, check_finite=False)
    sigma_inv_1 = cho_solve(cf, ones, check_finite=False)

    denom_mean = float(ones @ sigma_inv_1)
    ybar_g = float(np.mean(y_true)) if denom_mean <= 0 else float((ones @ sigma_inv_y) / denom_mean)

    centered = y_true - ybar_g
    gtss = float(centered.T @ cho_solve(cf, centered, check_finite=False))
    gr2 = np.nan if gtss <= 0 else 1.0 - grss / gtss

    return {"gr2": gr2}


def fit_log_poly_statsmodels(x, y_raw, order, cov_log):
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    y_log = np.log(y_raw)
    X = build_design_matrix_log_poly(x, order)
    sigma = nearest_psd(cov_log)
    model = sm.GLS(y_log, X, sigma=sigma)
    res = model.fit()

    params = np.asarray(res.params, dtype=float)
    yhat_log = np.asarray(res.predict(X), dtype=float)
    yhat_raw = np.exp(yhat_log)

    return {"params": params, "bic": float(res.bic), "yhat_raw": yhat_raw}


def bootstrap_refit_log_poly(x_fit, boot_y_raw, order, cov_log=None, n_use=None, random_state=42):
    boot_y_raw = np.asarray(boot_y_raw, dtype=float)
    if boot_y_raw.ndim != 2:
        raise ValueError("boot_y_raw must be 2D.")

    n_boot_total = boot_y_raw.shape[0]
    if n_boot_total == 0:
        return None

    n_use = pick_bootstrap_n_use(n_boot_total, n_use)
    idx = np.arange(n_boot_total)

    if n_use < n_boot_total:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, size=n_use, replace=False)

    params_list = []

    for i in idx:
        yb_raw = np.asarray(boot_y_raw[i], dtype=float)
        if np.any(~np.isfinite(yb_raw)) or np.any(yb_raw <= 0):
            continue
        fit_res = fit_log_poly_statsmodels(x=x_fit, y_raw=yb_raw, order=order, cov_log=cov_log)
        params_list.append(fit_res["params"])

    if len(params_list) == 0:
        return None

    params_arr = np.vstack(params_list)
    return {"n_boot_success": int(params_arr.shape[0]), "boot_params_arr": params_arr}


def _boot_std(x):
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1))


def _boot_ci(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    lo = float(np.percentile(x, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(x, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_param_inference(boot_params_arr, order, alpha=0.05):
    if boot_params_arr is None or len(boot_params_arr) == 0:
        return None

    boot_params_arr = np.asarray(boot_params_arr, dtype=float)
    out = {"n_boot_param_success": int(boot_params_arr.shape[0])}

    gamma_boot = boot_params_arr[:, 0]
    alpha_boot = np.exp(gamma_boot)

    out["gamma_boot_se"] = _boot_std(gamma_boot)
    out["gamma_boot_ci_low"], out["gamma_boot_ci_high"] = _boot_ci(gamma_boot, alpha=alpha)

    out["alpha_boot_se"] = _boot_std(alpha_boot)
    out["alpha_boot_ci_low"], out["alpha_boot_ci_high"] = _boot_ci(alpha_boot, alpha=alpha)

    for k in range(1, 3):
        if k <= order and k < boot_params_arr.shape[1]:
            beta_boot = boot_params_arr[:, k]
            out[f"beta{k}_boot_se"] = _boot_std(beta_boot)
            out[f"beta{k}_boot_ci_low"], out[f"beta{k}_boot_ci_high"] = _boot_ci(beta_boot, alpha=alpha)
        else:
            out[f"beta{k}_boot_se"] = np.nan
            out[f"beta{k}_boot_ci_low"] = np.nan
            out[f"beta{k}_boot_ci_high"] = np.nan

    out["beta_boot_se"] = out.get("beta1_boot_se", np.nan)
    out["beta_boot_ci_low"] = out.get("beta1_boot_ci_low", np.nan)
    out["beta_boot_ci_high"] = out.get("beta1_boot_ci_high", np.nan)

    return out


def extract_param_rows(fit_res, order, boot_inf=None):
    params = np.asarray(fit_res["params"], dtype=float)

    row = {"gamma": float(params[0]), "alpha": float(np.exp(params[0])), "gamma_boot_se": np.nan, "gamma_boot_ci_low": np.nan, "gamma_boot_ci_high": np.nan, "alpha_boot_se": np.nan, "alpha_boot_ci_low": np.nan, "alpha_boot_ci_high": np.nan, "beta_boot_se": np.nan, "beta_boot_ci_low": np.nan, "beta_boot_ci_high": np.nan, "n_boot_param_success": 0}

    for k in range(1, 3):
        if k <= order:
            row[f"beta{k}"] = float(params[k])
            row[f"beta{k}_boot_se"] = np.nan
            row[f"beta{k}_boot_ci_low"] = np.nan
            row[f"beta{k}_boot_ci_high"] = np.nan
        else:
            row[f"beta{k}"] = np.nan
            row[f"beta{k}_boot_se"] = np.nan
            row[f"beta{k}_boot_ci_low"] = np.nan
            row[f"beta{k}_boot_ci_high"] = np.nan

    if boot_inf is not None:
        row["n_boot_param_success"] = int(boot_inf.get("n_boot_param_success", 0))
        row["gamma_boot_se"] = boot_inf.get("gamma_boot_se", np.nan)
        row["gamma_boot_ci_low"] = boot_inf.get("gamma_boot_ci_low", np.nan)
        row["gamma_boot_ci_high"] = boot_inf.get("gamma_boot_ci_high", np.nan)

        row["alpha_boot_se"] = boot_inf.get("alpha_boot_se", np.nan)
        row["alpha_boot_ci_low"] = boot_inf.get("alpha_boot_ci_low", np.nan)
        row["alpha_boot_ci_high"] = boot_inf.get("alpha_boot_ci_high", np.nan)

        row["beta_boot_se"] = boot_inf.get("beta_boot_se", np.nan)
        row["beta_boot_ci_low"] = boot_inf.get("beta_boot_ci_low", np.nan)
        row["beta_boot_ci_high"] = boot_inf.get("beta_boot_ci_high", np.nan)

        for k in range(1, 3):
            row[f"beta{k}_boot_se"] = boot_inf.get(f"beta{k}_boot_se", np.nan)
            row[f"beta{k}_boot_ci_low"] = boot_inf.get(f"beta{k}_boot_ci_low", np.nan)
            row[f"beta{k}_boot_ci_high"] = boot_inf.get(f"beta{k}_boot_ci_high", np.nan)

    return row


def fit_one_stratum(row_meta, orders, min_delta_t, bootstrap_n_use, random_state, out_dir):
    npz_path = row_meta["npz_path"]
    stratum_name = row_meta["stratum_name"]

    npz_obj = np.load(npz_path, allow_pickle=True)

    delta_t_all = np.asarray(npz_obj["delta_t"], dtype=float)
    delta_y_all = np.asarray(npz_obj["delta_mcc"], dtype=float)

    mask = build_fit_mask(delta_t_all, delta_y_all, min_delta_t=min_delta_t)
    x_fit = delta_t_all[mask]
    y_raw = delta_y_all[mask]

    if len(x_fit) < 4:
        return []

    _, cov_log = get_formula_var_cov(npz_obj, mask, fit_space="log")
    _, cov_raw = get_formula_var_cov(npz_obj, mask, fit_space="raw")

    boot_y_raw = None
    if "boot_delta_mcc" in npz_obj.files:
        boot_y_raw = np.asarray(npz_obj["boot_delta_mcc"], dtype=float)[:, mask]
        row_keep = np.all(np.isfinite(boot_y_raw), axis=1) & np.all(boot_y_raw > 0, axis=1)
        boot_y_raw = boot_y_raw[row_keep]
        if boot_y_raw.shape[0] == 0:
            boot_y_raw = None

    fit_rows = []
    save_dict = {"stratum_name": np.array([stratum_name]), "x_fit": x_fit, "y_obs": y_raw, "cov_raw": cov_raw, "cov_log": cov_log}

    for order in orders:
        fit_res = fit_log_poly_statsmodels(x=x_fit, y_raw=y_raw, order=order, cov_log=cov_log)
        yhat_raw = fit_res["yhat_raw"]

        boot_refit = None
        boot_inf = None
        if boot_y_raw is not None:
            boot_refit = bootstrap_refit_log_poly(x_fit=x_fit, boot_y_raw=boot_y_raw, order=order, cov_log=cov_log, n_use=bootstrap_n_use, random_state=random_state)
            if boot_refit is not None:
                boot_inf = bootstrap_param_inference(boot_params_arr=boot_refit["boot_params_arr"], order=order, alpha=0.05)

        row = {"stratum_name": stratum_name, "npz_path": npz_path, "fit_space": "log", "method": "gls", "order": order, "n_fit_t": len(x_fit), "min_delta_t": float(np.min(x_fit)), "max_delta_t": float(np.max(x_fit)), "bic": fit_res["bic"]}
        row.update(extract_param_rows(fit_res, order, boot_inf=boot_inf))
        row.update(metrics_gls(y_raw, yhat_raw, cov_raw))
        fit_rows.append(row)

        key = f"gls_order{order}"
        save_dict[f"yhat_{key}"] = yhat_raw
        save_dict[f"params_{key}"] = np.asarray(fit_res["params"], dtype=float)
        if boot_refit is not None:
            save_dict[f"boot_params_arr_{key}"] = boot_refit["boot_params_arr"]

    detail_npz_path = os.path.join(out_dir, "fit_curves", f"{safe_name(stratum_name)}_fit_details.npz")
    ensure_dir(os.path.dirname(detail_npz_path))
    np.savez_compressed(detail_npz_path, **save_dict)

    for rr in fit_rows:
        for k, v in row_meta.items():
            if k not in rr:
                rr[k] = v
        rr["detail_npz_path"] = detail_npz_path

    return fit_rows


def build_side_by_side(df_fit):
    value_cols = ["alpha", "gamma", "beta1", "beta2", "beta_boot_se", "beta_boot_ci_low", "beta_boot_ci_high", "beta1_boot_se", "beta1_boot_ci_low", "beta1_boot_ci_high", "beta2_boot_se", "beta2_boot_ci_low", "beta2_boot_ci_high", "bic", "gr2"]
    value_cols = [c for c in value_cols if c in df_fit.columns]

    tmp = df_fit.copy()
    tmp["method_order"] = tmp["method"] + "_order" + tmp["order"].astype(str)

    side = tmp.pivot_table(index="stratum_name", columns="method_order", values=value_cols, aggfunc="first")
    side.columns = [f"{a}_{b}" for a, b in side.columns]
    return side.reset_index()


def summarize_by_method_order(fit_ok):
    rows = []
    for order in sorted(fit_ok["order"].unique()):
        sub = fit_ok[fit_ok["order"] == order].copy()
        if len(sub) == 0:
            continue

        row = {"method": "gls", "order": order, "n_strata": len(sub), "alpha_median": sub["alpha"].median() if "alpha" in sub else np.nan, "beta1_median": sub["beta1"].median() if "beta1" in sub else np.nan, "beta2_median": sub["beta2"].median() if "beta2" in sub else np.nan, "bic_median": sub["bic"].median() if "bic" in sub else np.nan, "gr2_median": sub["gr2"].median() if "gr2" in sub else np.nan}
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Fit GLS log-space polynomial models from saved MCC bootstrap npz files.")
    parser.add_argument("--index_csv", default="./results_boot/bootstrap_mcc_index.csv")
    parser.add_argument("--output", default="./results_fit_log_poly")
    parser.add_argument("--orders", default="1,2", help="all or comma-separated: 1,2")
    parser.add_argument("--min_delta_t", type=float, default=1.0)
    parser.add_argument("--bootstrap_n_use", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    orders = parse_orders(args.orders)
    orders_tag = "all" if args.orders.strip().lower() == "all" else "-".join([str(x) for x in orders])
    run_tag = f"log_poly__gls_order_{orders_tag}"
    out_dir = os.path.join(args.output, run_tag)

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "fit_curves"))

    meta_df = pd.read_csv(args.index_csv).copy()
    fit_rows_all = []

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Fitting strata"):
        row_meta = row.to_dict()
        fit_rows = fit_one_stratum(row_meta=row_meta, orders=orders, min_delta_t=args.min_delta_t, bootstrap_n_use=args.bootstrap_n_use, random_state=args.random_state, out_dir=out_dir)
        fit_rows_all.extend(fit_rows)

    fit_df = pd.DataFrame(fit_rows_all)
    fit_df.to_csv(os.path.join(out_dir, "fit_results_long.csv"), index=False)

    if len(fit_df) == 0:
        print("No successful fits.")
        return

    side_df = build_side_by_side(fit_df)
    meta_keep_cols = [c for c in meta_df.columns if c not in side_df.columns]
    merged = side_df.merge(meta_df[["stratum_name"] + meta_keep_cols], on="stratum_name", how="left")
    merged.to_csv(os.path.join(out_dir, "fit_results_side_by_side_with_meta.csv"), index=False)

    summary_df = summarize_by_method_order(fit_df)
    summary_df.to_csv(os.path.join(out_dir, "fit_summary_by_method_order.csv"), index=False)

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()