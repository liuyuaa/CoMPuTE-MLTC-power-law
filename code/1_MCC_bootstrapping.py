#!/usr/bin/env python3
import os
import ast
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm

COND_LIST = [
    "Anx", "Dep", "SMI", "Ast", "COPD", "Diab", "Hyp", "CHD", "StroTIA",
    "AF", "HF", "PAD", "CKD", "Dem", "Park", "Ost", "RA", "Can"
]
AGE_BANDS = [(20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79)]


def assign_age_band(df, bands, index_age_col="1st_age", band_label="1st_age_band"):
    def as_band(a):
        if pd.isna(a):
            return np.nan
        a = int(a)
        for lo, hi in bands:
            if lo <= a <= hi:
                return (lo, hi)
        return np.nan

    out = df.copy()
    out[band_label] = out[index_age_col].apply(as_band)
    return out


def riskset_over_grid(enter_age, exit_age, t_grid):
    enter_sorted = np.sort(np.asarray(enter_age, dtype=int))
    exit_sorted = np.sort(np.asarray(exit_age, dtype=int))
    entered_before = np.searchsorted(enter_sorted, t_grid, side="right")
    exited_before = np.searchsorted(exit_sorted, t_grid, side="left")
    return entered_before - exited_before


def compute_mcc_curve_yearscale(events_times, enter_age, exit_age, t_grid):
    events_times = np.asarray(events_times, dtype=int)
    enter_age = np.asarray(enter_age, dtype=int)
    exit_age = np.asarray(exit_age, dtype=int)
    t_grid = np.asarray(t_grid, dtype=int)

    y_grid = riskset_over_grid(enter_age, exit_age, t_grid).astype(float)

    if events_times.size == 0:
        z = np.zeros(len(t_grid), dtype=float)
        return pd.DataFrame({"t": t_grid, "Y": y_grid, "mcc": z.copy(), "mcc_var": z.copy(), "mcc_se": z.copy()})

    ev_count = pd.Series(events_times).value_counts().sort_index()
    ev_times = ev_count.index.to_numpy(dtype=int)
    dj = ev_count.to_numpy(dtype=float)

    enter_sorted = np.sort(enter_age)
    exit_sorted = np.sort(exit_age)
    entered_before = np.searchsorted(enter_sorted, ev_times, side="right")
    exited_before = np.searchsorted(exit_sorted, ev_times, side="left")
    yj = (entered_before - exited_before).astype(float)

    inc = np.zeros(len(ev_times), dtype=float)
    var_inc = np.zeros(len(ev_times), dtype=float)
    valid = yj > 0
    inc[valid] = dj[valid] / yj[valid]
    var_inc[valid] = dj[valid] / (yj[valid] ** 2)

    mcc_at_ev = np.cumsum(inc)
    var_at_ev = np.cumsum(var_inc)

    idx = np.searchsorted(ev_times, t_grid, side="right") - 1
    has_prev = idx >= 0
    safe_idx = np.clip(idx, 0, len(ev_times) - 1)

    mcc_grid = np.where(has_prev, mcc_at_ev[safe_idx], 0.0)
    var_grid = np.where(has_prev, var_at_ev[safe_idx], 0.0)

    return pd.DataFrame({"t": t_grid, "Y": y_grid, "mcc": mcc_grid, "mcc_var": var_grid, "mcc_se": np.sqrt(var_grid)})


def build_events_long(df, cond_cols, id_col, landmark_col_name="__lm__"):
    ev = df.melt(id_vars=[id_col, landmark_col_name], value_vars=cond_cols, var_name="condition", value_name="onset_age").dropna(subset=["onset_age"]).copy()
    ev["event_time"] = ev["onset_age"].astype(int)
    return ev[[id_col, "condition", "event_time"]]


def build_mcc_for_one_stratum(sub_people, cond_cols, t_grid, id_col, exit_col, min_alive=1000):
    tmp = sub_people.copy()
    tmp["__lm__"] = 0
    ev_long = build_events_long(tmp, cond_cols, id_col=id_col, landmark_col_name="__lm__")

    enter = np.zeros(len(sub_people), dtype=int)
    exit_ = sub_people[exit_col].astype(int).to_numpy()

    y = riskset_over_grid(enter, exit_, t_grid)
    valid_idx = np.where(y >= min_alive)[0]
    if len(valid_idx) == 0:
        return None

    max_valid_t = int(t_grid[valid_idx[-1]])
    grid_trunc = t_grid[t_grid <= max_valid_t]

    out = compute_mcc_curve_yearscale(ev_long["event_time"].to_numpy(), enter, exit_, grid_trunc)
    out["n_people"] = sub_people[id_col].nunique()
    return out


def build_mcc_for_one_stratum_on_fixed_grid(sub_people, cond_cols, fixed_t_grid, id_col, exit_col):
    tmp = sub_people.copy()
    tmp["__lm__"] = 0
    ev_long = build_events_long(tmp, cond_cols, id_col=id_col, landmark_col_name="__lm__")

    enter = np.zeros(len(sub_people), dtype=int)
    exit_ = sub_people[exit_col].astype(int).to_numpy()

    out = compute_mcc_curve_yearscale(ev_long["event_time"].to_numpy(), enter, exit_, np.asarray(fixed_t_grid, dtype=int))
    out["n_people"] = sub_people[id_col].nunique()
    return out


def bootstrap_curve_worker(args):
    sub_people, cond_cols, fixed_t_grid, id_col, exit_col, band_str, seed = args
    rng = np.random.default_rng(seed)

    ids = sub_people[id_col].unique()
    samp_ids = rng.choice(ids, size=len(ids), replace=True)
    counts = pd.Series(samp_ids).value_counts()

    samp = sub_people[sub_people[id_col].isin(counts.index)].copy()
    samp["__boot_n__"] = samp[id_col].map(counts).astype(int)
    samp = samp.loc[samp.index.repeat(samp["__boot_n__"])].copy().reset_index(drop=True)
    samp = samp.drop(columns="__boot_n__")

    boot_curve = build_mcc_for_one_stratum_on_fixed_grid(samp, cond_cols, fixed_t_grid, id_col, exit_col)
    if boot_curve is None:
        return None

    band = ast.literal_eval(str(band_str))
    t_ref = int(band[1]) + 1
    if t_ref not in set(boot_curve["t"].astype(int).to_numpy()):
        return None

    mcc_ref = float(boot_curve.loc[boot_curve["t"].astype(int) == t_ref, "mcc"].iloc[0])
    delta = boot_curve["mcc"].astype(float).to_numpy() - mcc_ref

    return {"mcc": boot_curve["mcc"].astype(float).to_numpy(), "delta_mcc": delta, "mcc_ref": mcc_ref}


def format_gender(gender):
    g_map = {"Female": "F", "F": "F", "Male": "M", "M": "M"}
    return g_map.get(str(gender), str(gender))


def make_stratum_name(strata_info):
    gender = format_gender(strata_info["gender"])
    cond = str(strata_info["1st_cond"])
    band = ast.literal_eval(str(strata_info["1st_age_band"])) if isinstance(strata_info["1st_age_band"], str) else strata_info["1st_age_band"]
    age1, age2 = int(band[0]), int(band[1])
    return f"{gender}_{cond}_{age1}_{age2}"


def process_one_stratum(sub_people, cond_cols, t_grid, id_col, exit_col, band_str, strata_info, min_alive=1000, n_boot=100, n_jobs=None, random_state=42, out_dir=None):
    point_curve = build_mcc_for_one_stratum(sub_people=sub_people, cond_cols=cond_cols, t_grid=t_grid, id_col=id_col, exit_col=exit_col, min_alive=min_alive)
    if point_curve is None:
        return None

    fixed_t_grid = point_curve["t"].astype(int).to_numpy()
    band = ast.literal_eval(str(band_str))
    t_ref = int(band[1]) + 1

    ref_rows = point_curve.loc[point_curve["t"].astype(int) == t_ref]
    if ref_rows.empty:
        return None

    ref_row = ref_rows.iloc[0]
    mcc_ref = float(ref_row["mcc"])
    var_ref = float(ref_row["mcc_var"])

    point_curve["t_ref"] = t_ref
    point_curve["mcc_ref"] = mcc_ref
    point_curve["var_ref"] = var_ref
    point_curve["delta_t"] = point_curve["t"].astype(int) - t_ref
    point_curve["delta_mcc"] = point_curve["mcc"].astype(float) - mcc_ref
    point_curve["var_delta_mcc"] = np.maximum(point_curve["mcc_var"].astype(float) - var_ref, 0.0)
    point_curve["se_delta_mcc"] = np.sqrt(point_curve["var_delta_mcc"])

    seeds = np.random.default_rng(random_state).integers(0, 2**31 - 1, size=n_boot)
    args_list = [
        (sub_people, cond_cols, fixed_t_grid, id_col, exit_col, band_str, int(seed))
        for seed in seeds
    ]

    n_jobs_use = n_jobs or cpu_count()
    with Pool(processes=n_jobs_use) as pool:
        boot_results = list(
            tqdm(
                pool.imap(bootstrap_curve_worker, args_list),
                total=n_boot,
                desc=f"Bootstrap {strata_info}",
            )
        )

    boot_results = [x for x in boot_results if x is not None]
    if len(boot_results) == 0:
        return None

    boot_mcc = np.vstack([x["mcc"] for x in boot_results])
    boot_delta = np.vstack([x["delta_mcc"] for x in boot_results])

    point_curve["mcc_lo"] = np.percentile(boot_mcc, 2.5, axis=0)
    point_curve["mcc_hi"] = np.percentile(boot_mcc, 97.5, axis=0)
    point_curve["delta_mcc_lo"] = np.percentile(boot_delta, 2.5, axis=0)
    point_curve["delta_mcc_hi"] = np.percentile(boot_delta, 97.5, axis=0)

    for k, v in strata_info.items():
        point_curve[k] = v

    stratum_name = make_stratum_name(strata_info)
    npz_path = os.path.join(out_dir, f"{stratum_name}.npz")

    np.savez_compressed(
        npz_path,
        t=point_curve["t"].to_numpy(),
        Y=point_curve["Y"].to_numpy(),
        mcc=point_curve["mcc"].to_numpy(),
        mcc_var=point_curve["mcc_var"].to_numpy(),
        mcc_se=point_curve["mcc_se"].to_numpy(),
        n_people=np.array([point_curve["n_people"].iloc[0]]),
        t_ref=np.array([t_ref]),
        mcc_ref=np.array([mcc_ref]),
        var_ref=np.array([var_ref]),
        delta_t=point_curve["delta_t"].to_numpy(),
        delta_mcc=point_curve["delta_mcc"].to_numpy(),
        var_delta_mcc=point_curve["var_delta_mcc"].to_numpy(),
        se_delta_mcc=point_curve["se_delta_mcc"].to_numpy(),
        mcc_lo=point_curve["mcc_lo"].to_numpy(),
        mcc_hi=point_curve["mcc_hi"].to_numpy(),
        delta_mcc_lo=point_curve["delta_mcc_lo"].to_numpy(),
        delta_mcc_hi=point_curve["delta_mcc_hi"].to_numpy(),
        boot_mcc=boot_mcc,
        boot_delta_mcc=boot_delta,
    )

    return {**strata_info, "stratum_name": stratum_name, "npz_path": npz_path, "n_boot_success": len(boot_results), "n_t": len(point_curve), "t_ref": t_ref, "mcc_ref": mcc_ref, "var_ref": var_ref, "n_people": int(point_curve["n_people"].iloc[0])}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build bootstrap MCC curves and save reusable results.")
    parser.add_argument("--input", default="./data/df_patient_20_80.csv")
    parser.add_argument("--output", default="./results_boot/")
    parser.add_argument("--id_col", default="patid")
    parser.add_argument("--index_age_col", default="1st_age")
    parser.add_argument("--exit_col", default="exit_age")
    parser.add_argument("--min_alive", type=int, default=1000)
    parser.add_argument("--max_t", type=int, default=100)
    parser.add_argument("--n_boot", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Reading input file from {args.input}")
    df = pd.read_csv(args.input)

    cond_cols = COND_LIST
    df.loc[:, cond_cols] = (df[cond_cols].apply(pd.to_datetime, errors="coerce").apply(lambda x: x.dt.year).sub(df["yob"], axis=0).astype("Int64"))

    df = assign_age_band(df, AGE_BANDS, index_age_col=args.index_age_col, band_label="1st_age_band")
    df = df[df["1st_cond"] != "Health"].copy()
    df = df[df["1st_age_band"].notna()].copy()

    print("Assigning age bands and filtering...")
    t_grid = np.arange(0, args.max_t + 1, dtype=int)
    strata_cols = ["gender", "1st_cond", "1st_age_band"]

    print("Building bootstrap MCC curves...")
    meta_rows = []
    grouped = df.groupby(strata_cols, dropna=False, sort=True)

    for key_vals, sub in tqdm(grouped, total=grouped.ngroups, desc="Build bootstrap MCC"):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        if pd.isna(key_vals[-1]):
            continue

        strata_info = {c: (str(v) if isinstance(v, (tuple, list)) else v) for c, v in zip(strata_cols, key_vals)}
        band_str = str(strata_info["1st_age_band"])

        row = process_one_stratum(
            sub_people=sub, cond_cols=cond_cols, t_grid=t_grid, id_col=args.id_col, exit_col=args.exit_col, band_str=band_str, strata_info=strata_info, min_alive=args.min_alive, n_boot=args.n_boot, n_jobs=args.n_jobs, random_state=args.random_state, out_dir=args.output)
        if row is not None:
            meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)
    out_csv = os.path.join(args.output, "bootstrap_mcc_index.csv")
    meta_df.to_csv(out_csv, index=False)
    print(f"Saved index to {out_csv}")