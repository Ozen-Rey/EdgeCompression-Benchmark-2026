"""
BD-3D Metrics v3: estensione triassiale rigorosa della Bjontegaard Distortion-rate.
Gestione integrata della colonna 'preset' per supportare varianti multiple dello stesso codec.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RBFInterpolator
import warnings

# ============================================================
# Utility: bd_result type per metriche con possibile ND
# ============================================================


def bd_result(value, valid=True, reason=""):
    return {
        "value": float(value) if value is not None else float("nan"),
        "valid": valid,
        "reason": reason,
    }


# ============================================================
# 1) BD-rate classica (1D)
# ============================================================


def bd_rate_classic(rates_a, dist_a, rates_b, dist_b, method="cubic"):
    rates_a, dist_a = np.asarray(rates_a), np.asarray(dist_a)
    rates_b, dist_b = np.asarray(rates_b), np.asarray(dist_b)

    if len(rates_a) < 4 or len(rates_b) < 4:
        return bd_result(np.nan, False, "less than 4 points")

    log_r_a = np.log10(rates_a)
    log_r_b = np.log10(rates_b)

    sa = np.argsort(dist_a)
    sb = np.argsort(dist_b)
    dist_a, log_r_a = dist_a[sa], log_r_a[sa]
    dist_b, log_r_b = dist_b[sb], log_r_b[sb]

    d_min = max(dist_a.min(), dist_b.min())
    d_max = min(dist_a.max(), dist_b.max())
    if d_max <= d_min:
        return bd_result(np.nan, False, f"distortion overlap empty")

    a_range = dist_a.max() - dist_a.min()
    b_range = dist_b.max() - dist_b.min()
    overlap_size = d_max - d_min
    if overlap_size < 0.3 * min(a_range, b_range):
        return bd_result(np.nan, False, f"distortion overlap too small")

    f_a = interp1d(dist_a, log_r_a, kind=method, bounds_error=False)
    f_b = interp1d(dist_b, log_r_b, kind=method, bounds_error=False)

    d_grid = np.linspace(d_min, d_max, 1000)
    delta_log_r = np.mean(f_b(d_grid) - f_a(d_grid))

    return bd_result((10**delta_log_r - 1) * 100, True, "")


# ============================================================
# 2) BD-rate-E (2D condizionata) in log-energy space
# ============================================================


def bd_rate_e_conditional(df_a, df_b, n_energy_bins=10, min_window=4, method="cubic"):
    rates_a = df_a["rate_mbps"].values
    dist_a = df_a["distortion"].values
    energy_a = df_a["energy_j"].values
    rates_b = df_b["rate_mbps"].values
    dist_b = df_b["distortion"].values
    energy_b = df_b["energy_j"].values

    if len(rates_a) < min_window or len(rates_b) < min_window:
        return {**bd_result(np.nan, False, "insufficient points"), "n_valid_bins": 0}

    log_e_a = np.log10(energy_a)
    log_e_b = np.log10(energy_b)

    le_min = max(log_e_a.min(), log_e_b.min())
    le_max = min(log_e_a.max(), log_e_b.max())

    if le_max <= le_min:
        return {
            **bd_result(np.nan, False, "log-energy overlap empty"),
            "n_valid_bins": 0,
        }

    overlap_decades = le_max - le_min
    if overlap_decades < 0.5:
        return {
            **bd_result(np.nan, False, "log-energy overlap too narrow"),
            "n_valid_bins": 0,
        }

    energy_grid = np.linspace(le_min, le_max, n_energy_bins)
    bd_rates = []

    for log_e_star in energy_grid:
        order_a = np.argsort(np.abs(log_e_a - log_e_star))[:min_window]
        order_b = np.argsort(np.abs(log_e_b - log_e_star))[:min_window]
        if len(order_a) < min_window or len(order_b) < min_window:
            continue
        bd = bd_rate_classic(
            rates_a[order_a],
            dist_a[order_a],
            rates_b[order_b],
            dist_b[order_b],
            method=method,
        )
        if bd["valid"] and not np.isnan(bd["value"]):
            bd_rates.append(bd["value"])

    if len(bd_rates) == 0:
        return {**bd_result(np.nan, False, "no valid energy bins"), "n_valid_bins": 0}

    return {**bd_result(np.mean(bd_rates), True, ""), "n_valid_bins": len(bd_rates)}


# ============================================================
# 3) BD-volume (3D piena) in log-energy space
# ============================================================


def bd_volume(df_a, df_b, n_grid=30, min_overlap_decades=0.5):
    rates_a = df_a["rate_mbps"].values
    dist_a = df_a["distortion"].values
    energy_a = df_a["energy_j"].values
    rates_b = df_b["rate_mbps"].values
    dist_b = df_b["distortion"].values
    energy_b = df_b["energy_j"].values

    if len(rates_a) < 6 or len(rates_b) < 6:
        return {
            **bd_result(np.nan, False, "less than 6 points"),
            "overlap_d": 0,
            "overlap_e_dec": 0,
        }

    log_e_a = np.log10(energy_a)
    log_e_b = np.log10(energy_b)

    d_min = max(dist_a.min(), dist_b.min())
    d_max = min(dist_a.max(), dist_b.max())
    le_min = max(log_e_a.min(), log_e_b.min())
    le_max = min(log_e_a.max(), log_e_b.max())

    overlap_d = max(0, d_max - d_min)
    overlap_e_dec = max(0, le_max - le_min)

    a_d_range = dist_a.max() - dist_a.min()
    b_d_range = dist_b.max() - dist_b.min()

    if overlap_d < 0.3 * min(a_d_range, b_d_range):
        return {
            **bd_result(np.nan, False, "distortion overlap too small"),
            "overlap_d": overlap_d,
            "overlap_e_dec": overlap_e_dec,
        }

    if overlap_e_dec < min_overlap_decades:
        return {
            **bd_result(np.nan, False, "log-energy overlap too narrow"),
            "overlap_d": overlap_d,
            "overlap_e_dec": overlap_e_dec,
        }

    points_a = np.column_stack([dist_a, log_e_a])
    points_b = np.column_stack([dist_b, log_e_b])
    log_r_a = np.log10(rates_a)
    log_r_b = np.log10(rates_b)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rbf_a = RBFInterpolator(
                points_a, log_r_a, kernel="thin_plate_spline", smoothing=0.1
            )
            rbf_b = RBFInterpolator(
                points_b, log_r_b, kernel="thin_plate_spline", smoothing=0.1
            )
    except Exception as e:
        return {
            **bd_result(np.nan, False, f"RBF fit failed"),
            "overlap_d": overlap_d,
            "overlap_e_dec": overlap_e_dec,
        }

    d_grid = np.linspace(d_min, d_max, n_grid)
    le_grid = np.linspace(le_min, le_max, n_grid)
    DD, EE = np.meshgrid(d_grid, le_grid)
    eval_points = np.column_stack([DD.ravel(), EE.ravel()])

    z_a = rbf_a(eval_points).reshape(n_grid, n_grid)
    z_b = rbf_b(eval_points).reshape(n_grid, n_grid)

    delta_log_r = z_b - z_a
    bd_log = np.trapz(np.trapz(delta_log_r, le_grid, axis=0), d_grid)
    bd_log /= overlap_d * overlap_e_dec

    return {
        **bd_result((10**bd_log - 1) * 100, True, ""),
        "overlap_d": overlap_d,
        "overlap_e_dec": overlap_e_dec,
    }


# ============================================================
# 4) Pareto-HV indicator e 5) Epsilon-indicator
# ============================================================


def compute_pareto_front(points, directions=("min", "max", "min")):
    N = len(points)
    is_pareto = np.ones(N, dtype=bool)
    pts = points.copy()
    for i, d in enumerate(directions):
        if d == "max":
            pts[:, i] = -pts[:, i]
    for i in range(N):
        if not is_pareto[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                is_pareto[i] = False
                break
    return is_pareto


def hypervolume_mc(
    points, ref_point, directions=("min", "max", "min"), n_samples=10000, seed=42
):
    pareto_mask = compute_pareto_front(points, directions)
    pareto_pts = points[pareto_mask]
    if len(pareto_pts) == 0:
        return 0.0

    box_min = np.minimum(pareto_pts.min(axis=0), ref_point)
    box_max = np.maximum(pareto_pts.max(axis=0), ref_point)
    box_volume = float(np.prod(box_max - box_min))
    if box_volume == 0:
        return 0.0

    rng = np.random.default_rng(seed=seed)
    samples = rng.uniform(box_min, box_max, size=(n_samples, 3))

    pareto_signed, samples_signed, ref_signed = (
        pareto_pts.copy(),
        samples.copy(),
        ref_point.copy(),
    )
    for i, d in enumerate(directions):
        if d == "max":
            pareto_signed[:, i] = -pareto_signed[:, i]
            samples_signed[:, i] = -samples_signed[:, i]
            ref_signed[i] = -ref_signed[i]

    n_dominated = sum(
        1
        for s in samples_signed
        if np.all(s <= ref_signed) and np.any(np.all(pareto_signed <= s, axis=1))
    )
    return (n_dominated / n_samples) * box_volume


def pareto_hv_difference(
    df_a, df_b, global_ref_point, distortion_col="distortion", higher_is_better=True
):
    pts_a = np.column_stack(
        [
            np.log10(df_a["rate_mbps"].values),
            df_a[distortion_col].values,
            np.log10(df_a["energy_j"].values),
        ]
    )
    pts_b = np.column_stack(
        [
            np.log10(df_b["rate_mbps"].values),
            df_b[distortion_col].values,
            np.log10(df_b["energy_j"].values),
        ]
    )

    if len(pts_a) == 0 or len(pts_b) == 0:
        return bd_result(0.0, False, "empty fronts")

    directions = ("min", "max" if higher_is_better else "min", "min")
    hv_a = hypervolume_mc(pts_a, global_ref_point, directions=directions)
    hv_b = hypervolume_mc(pts_b, global_ref_point, directions=directions)

    if hv_a + hv_b == 0:
        return bd_result(0.0, False, "both HV=0")
    return bd_result((hv_b - hv_a) / max(hv_a, hv_b), True, "")


def epsilon_indicator(
    df_a,
    df_b,
    global_min,
    global_ranges,
    distortion_col="distortion",
    higher_is_better=True,
):
    pts_a = np.column_stack(
        [
            np.log10(df_a["rate_mbps"].values),
            df_a[distortion_col].values,
            np.log10(df_a["energy_j"].values),
        ]
    )
    pts_b = np.column_stack(
        [
            np.log10(df_b["rate_mbps"].values),
            df_b[distortion_col].values,
            np.log10(df_b["energy_j"].values),
        ]
    )

    if len(pts_a) == 0 or len(pts_b) == 0:
        return bd_result(np.nan, False, "empty fronts")

    safe_ranges = np.where(global_ranges == 0, 1, global_ranges)
    pts_a_n = (pts_a - global_min) / safe_ranges
    pts_b_n = (pts_b - global_min) / safe_ranges

    if higher_is_better:
        pts_a_n[:, 1] = 1 - pts_a_n[:, 1]
        pts_b_n[:, 1] = 1 - pts_b_n[:, 1]

    eps_max = -np.inf
    for a in pts_a_n:
        eps_min_b = np.min([np.max(b - a) for b in pts_b_n])
        if eps_min_b > eps_max:
            eps_max = eps_min_b

    return bd_result(float(eps_max), True, "")


def compute_seq_global_bounds(
    df_seq, distortion_col="distortion", higher_is_better=True
):
    log_r = np.log10(df_seq["rate_mbps"].values)
    dist = df_seq[distortion_col].values
    log_e = np.log10(df_seq["energy_j"].values)

    if higher_is_better:
        global_ref_point = np.array([log_r.max(), dist.min(), log_e.max()])
    else:
        global_ref_point = np.array([log_r.max(), dist.max(), log_e.max()])

    global_min = np.array([log_r.min(), dist.min(), log_e.min()])
    global_max = np.array([log_r.max(), dist.max(), log_e.max()])
    global_ranges = np.where((global_max - global_min) == 0, 1, global_max - global_min)

    return {
        "global_ref_point": global_ref_point,
        "global_min": global_min,
        "global_max": global_max,
        "global_ranges": global_ranges,
    }


# ============================================================
# Funzione di alto livello: matrix N×N pairwise aggiornata
# ============================================================


def compute_bd_matrix(
    energy_csv_path,
    quality_csv_path,
    distortion_metric="vmaf_mean",
    profile_filter="LDP",
    higher_is_better=True,
):
    df_e = pd.read_csv(energy_csv_path)
    df_q = pd.read_csv(quality_csv_path)

    # Standardizzazione colonna preset
    if "preset" not in df_e.columns:
        df_e["preset"] = "none"
    if "preset" not in df_q.columns:
        df_q["preset"] = "none"
    df_e["preset"] = df_e["preset"].fillna("none")
    df_q["preset"] = df_q["preset"].fillna("none")

    if "phase" in df_e.columns:
        df_e_total = (
            df_e.groupby(["codec", "seq", "profile", "preset", "param"])
            .agg({"energy_total_j": "sum", "time_s": "sum", "actual_mbps": "first"})
            .reset_index()
        )
    else:
        df_e_total = df_e

    df = pd.merge(df_e_total, df_q, on=["codec", "seq", "profile", "preset", "param"])
    df = df[df["profile"] == profile_filter].copy()
    df = df.rename(
        columns={
            "actual_mbps": "rate_mbps",
            "energy_total_j": "energy_j",
            distortion_metric: "distortion",
        }
    )

    # Creazione identificativo univoco (codec_variant)
    df["codec_variant"] = df.apply(
        lambda r: (
            r["codec"] if r["preset"] == "none" else f"{r['codec']}-{r['preset']}"
        ),
        axis=1,
    )

    variants = sorted(df["codec_variant"].unique())
    sequences = sorted(df["seq"].unique())
    rows = []

    for seq in sequences:
        df_seq = df[df["seq"] == seq]
        if len(df_seq) == 0:
            continue
        bounds = compute_seq_global_bounds(df_seq, "distortion", higher_is_better)

        for var_a in variants:
            df_a = df_seq[df_seq["codec_variant"] == var_a]
            if len(df_a) < 4:
                continue
            for var_b in variants:
                if var_a == var_b:
                    continue
                df_b = df_seq[df_seq["codec_variant"] == var_b]
                if len(df_b) < 4:
                    continue

                row = {"codec_a": var_a, "codec_b": var_b, "seq": seq}

                bd1 = bd_rate_classic(
                    df_a["rate_mbps"].values,
                    df_a["distortion"].values,
                    df_b["rate_mbps"].values,
                    df_b["distortion"].values,
                )
                row["bd_rate"], row["bd_rate_valid"] = bd1["value"], bd1["valid"]

                bd2 = bd_rate_e_conditional(df_a, df_b)
                row["bd_rate_e"], row["bd_rate_e_valid"], row["bd_rate_e_bins"] = (
                    bd2["value"],
                    bd2["valid"],
                    bd2.get("n_valid_bins", 0),
                )

                bd3 = bd_volume(df_a, df_b)
                row["bd_volume"], row["bd_volume_valid"] = bd3["value"], bd3["valid"]
                row["overlap_d"], row["overlap_e_dec"] = bd3.get(
                    "overlap_d", 0
                ), bd3.get("overlap_e_dec", 0)

                bd4 = pareto_hv_difference(
                    df_a,
                    df_b,
                    global_ref_point=bounds["global_ref_point"],
                    higher_is_better=higher_is_better,
                )
                row["pareto_hv"], row["pareto_hv_valid"] = bd4["value"], bd4["valid"]

                bd5 = epsilon_indicator(
                    df_a,
                    df_b,
                    global_min=bounds["global_min"],
                    global_ranges=bounds["global_ranges"],
                    higher_is_better=higher_is_better,
                )
                row["epsilon"], row["epsilon_valid"] = bd5["value"], bd5["valid"]

                rows.append(row)

    return pd.DataFrame(rows)
