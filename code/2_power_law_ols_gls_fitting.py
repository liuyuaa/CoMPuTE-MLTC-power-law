#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

METHODS_ALL = ["ols", "gls"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def parse_methods(s):
    s = s.strip().lower()
    if s == "all":
        return METHODS_ALL
    vals = [x.strip() for x in s.split(",") if x.strip()]
    for v in vals:
        if v not in METHODS_ALL:
            raise ValueError(f"Unknown method: {v}")
    return vals


def nearest_psd(mat, jitter=1e-10):
    mat = np.asarray(mat, dtype=float)
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.maximum(vals, jitter)
    out = vecs @ np.diag(vals) @ vecs.T
    return 0.5 * (out + out.T)


def model_raw(x, alpha, beta):
    x = np.asarray(x, dtype=float)
    return alpha * np.power(x, beta)


def model_log(x, alpha, beta):
    x = np.asarray(x, dtype=float)
    return np.log(alpha) + beta * np.log(x)


def predict_on_raw_scale(x, alpha, beta, fit_space):
    x = np.asarray(x, dtype=float)
    if fit_space == "raw":
        return model_raw(x, alpha, beta)
    return np.exp(model_log(x, alpha, beta))


def initial_guess(x, y_raw):
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    pos = (x > 0) & (y_raw > 0)
    if np.sum(pos) >= 2:
        beta0, log_alpha0 = np.polyfit(np.log(x[pos]), np.log(y_raw[pos]), 1)
        return np.array([np.exp(log_alpha0), beta0], dtype=float)

    return np.array([max(np.nanmedian(y_raw), 1e-8), 1.0], dtype=float)


def build_fit_mask(delta_t, delta_y, fit_space="raw", min_delta_t=1.0):
    delta_t = np.asarray(delta_t, dtype=float)
    delta_y = np.asarray(delta_y, dtype=float)

    mask = np.isfinite(delta_t) & np.isfinite(delta_y)
    mask &= delta_t >= min_delta_t

    if fit_space == "log":
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
        return y_raw, var_raw, cov_raw

    denom = np.outer(np.maximum(y_raw, 1e-12), np.maximum(y_raw, 1e-12))
    cov_log = cov_raw / denom
    cov_log = nearest_psd(cov_log)
    var_log = np.maximum(np.diag(cov_log), 1e-12)
    return y_raw, var_log, cov_log


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


def metrics_ols(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    sse = float(np.sum(resid ** 2))
    ybar = float(np.mean(y_true))
    sst = float(np.sum((y_true - ybar) ** 2))
    r2 = np.nan if sst <= 0 else 1.0 - sse / sst
    return {"r2": r2}


def metrics_gls(y_true, y_pred, cov_raw):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    n = len(y_true)

    grss, _, cf = generalized_ss(resid, cov_raw)

    ones = np.ones(n, dtype=float)
    sigma_inv_y = cho_solve(cf, y_true, check_finite=False)
    sigma_inv_1 = cho_solve(cf, ones, check_finite=False)

    denom_mean = float(ones @ sigma_inv_1)
    ybar_g = float(np.mean(y_true)) if denom_mean <= 0 else float((ones @ sigma_inv_y) / denom_mean)

    centered = y_true - ybar_g
    gtss = float(centered.T @ cho_solve(cf, centered, check_finite=False))
    gr2 = np.nan if gtss <= 0 else 1.0 - grss / gtss
    return {"gr2": gr2}


def objective_factory(x, y_fitspace, fit_space, method, cov_fit=None):
    x = np.asarray(x, dtype=float)
    y_fitspace = np.asarray(y_fitspace, dtype=float)

    if fit_space == "raw":
        def pred(theta):
            return model_raw(x, theta[0], theta[1])
    else:
        def pred(theta):
            return model_log(x, theta[0], theta[1])

    if method == "ols":
        def obj(theta):
            alpha, _ = theta
            if alpha <= 0:
                return np.inf
            r = y_fitspace - pred(theta)
            return float(np.sum(r ** 2))
        return obj

    if method == "gls":
        cf, _ = robust_cholesky(cov_fit)

        def obj(theta):
            alpha, _ = theta
            if alpha <= 0:
                return np.inf
            r = y_fitspace - pred(theta)
            return float(r.T @ cho_solve(cf, r, check_finite=False))
        return obj

    raise ValueError(f"Unknown method: {method}")


def fit_one_method(x, y_raw, fit_space, method, cov_fit=None, init_theta=None, maxiter=3000):
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    y_fitspace = y_raw if fit_space == "raw" else np.log(y_raw)

    if init_theta is None:
        init_theta = initial_guess(x, y_raw)

    obj = objective_factory(x=x, y_fitspace=y_fitspace, fit_space=fit_space, method=method, cov_fit=cov_fit)

    bounds = [(1e-12, None), (None, None)]
    res = minimize(obj, x0=np.asarray(init_theta, dtype=float), method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})

    alpha_hat, beta_hat = float(res.x[0]), float(res.x[1])
    yhat_raw = predict_on_raw_scale(x, alpha_hat, beta_hat, fit_space)

    return {"alpha": alpha_hat, "beta": beta_hat, "objective": float(res.fun), "yhat_raw": yhat_raw}


def fit_one_stratum(row_meta, fit_space, methods, min_delta_t, out_dir):
    npz_path = row_meta["npz_path"]
    stratum_name = row_meta["stratum_name"]

    npz_obj = np.load(npz_path, allow_pickle=True)

    delta_t_all = np.asarray(npz_obj["delta_t"], dtype=float)
    delta_y_all = np.asarray(npz_obj["delta_mcc"], dtype=float)

    mask = build_fit_mask(delta_t_all, delta_y_all, fit_space=fit_space, min_delta_t=min_delta_t)
    x_fit = delta_t_all[mask]

    if len(x_fit) < 4:
        return []

    y_raw, _, cov_raw = get_formula_var_cov(npz_obj, mask, fit_space="raw")
    _, _, cov_fit = get_formula_var_cov(npz_obj, mask, fit_space=fit_space)
    common_init = initial_guess(x_fit, y_raw)

    fit_rows = []
    curve_outputs = {}

    for method in methods:
        fit_res = fit_one_method(x=x_fit, y_raw=y_raw, fit_space=fit_space, method=method, cov_fit=cov_fit if method == "gls" else None, init_theta=common_init)

        yhat_raw = fit_res["yhat_raw"]
        row = {"stratum_name": stratum_name, "npz_path": npz_path, "fit_space": fit_space, "method": method, "n_fit_t": len(x_fit), "min_delta_t": float(np.min(x_fit)), "max_delta_t": float(np.max(x_fit)), "alpha": fit_res["alpha"], "beta": fit_res["beta"], "objective": fit_res["objective"]}

        if method == "ols":
            row.update(metrics_ols(y_raw, yhat_raw))
        elif method == "gls":
            row.update(metrics_gls(y_raw, yhat_raw, cov_raw))

        fit_rows.append(row)
        curve_outputs[method] = {"x_fit": x_fit, "y_obs": y_raw, "yhat_raw": yhat_raw, "cov_raw": cov_raw}

    detail_npz_path = os.path.join(out_dir, "fit_curves", f"{safe_name(stratum_name)}_fit_details.npz")
    ensure_dir(os.path.dirname(detail_npz_path))

    save_dict = {"stratum_name": np.array([stratum_name]), "x_fit": x_fit, "y_obs": y_raw, "cov_raw": cov_raw}

    for method in methods:
        if method in curve_outputs:
            save_dict[f"yhat_{method}"] = curve_outputs[method]["yhat_raw"]

    np.savez_compressed(detail_npz_path, **save_dict)

    for rr in fit_rows:
        for k, v in row_meta.items():
            if k not in rr:
                rr[k] = v
        rr["detail_npz_path"] = detail_npz_path

    return fit_rows


def build_side_by_side(df_fit):
    value_cols = [c for c in ["alpha", "beta", "r2", "gr2"] if c in df_fit.columns]
    side = df_fit.pivot_table(index="stratum_name", columns="method", values=value_cols, aggfunc="first")
    side.columns = [f"{a}_{b}" for a, b in side.columns]
    return side.reset_index()


def summarize_by_method(fit_ok):
    rows = []
    for method in sorted(fit_ok["method"].unique()):
        sub = fit_ok[fit_ok["method"] == method].copy()
        row = {"method": method, "n_strata": len(sub), "alpha_median": sub["alpha"].median(), "beta_median": sub["beta"].median()}
        if method == "ols" and "r2" in sub.columns:
            row["r2_median"] = sub["r2"].median()
        elif method == "gls" and "gr2" in sub.columns:
            row["gr2_median"] = sub["gr2"].median()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Fit power-law curves from saved MCC npz files.")
    parser.add_argument("--index_csv", default="./results_boot/bootstrap_mcc_index.csv")
    parser.add_argument("--output", default="./results_fit")
    parser.add_argument("--fit_space", choices=["raw", "log"], default="raw")
    parser.add_argument("--methods", default="all", help="all or comma-separated: ols,gls")
    parser.add_argument("--min_delta_t", type=float, default=1.0)
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    methods_tag = "all" if args.methods.strip().lower() == "all" else "-".join(methods)
    run_tag = f"{args.fit_space}__{methods_tag}"
    out_dir = os.path.join(args.output, run_tag)

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "fit_curves"))

    meta_df = pd.read_csv(args.index_csv).copy()
    fit_rows_all = []

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Fitting strata"):
        row_meta = row.to_dict()
        fit_rows = fit_one_stratum(row_meta=row_meta, fit_space=args.fit_space, methods=methods, min_delta_t=args.min_delta_t, out_dir=out_dir)
        fit_rows_all.extend(fit_rows)

    fit_df = pd.DataFrame(fit_rows_all)
    fit_df.to_csv(os.path.join(out_dir, "fit_results_long.csv"), index=False)

    side_df = build_side_by_side(fit_df)
    meta_keep_cols = [c for c in meta_df.columns if c not in side_df.columns]
    merged = side_df.merge(meta_df[["stratum_name"] + meta_keep_cols], on="stratum_name", how="left")
    merged.to_csv(os.path.join(out_dir, "fit_results_side_by_side_with_meta.csv"), index=False)

    summary_df = summarize_by_method(fit_df)
    summary_df.to_csv(os.path.join(out_dir, "fit_summary_by_method.csv"), index=False)

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()