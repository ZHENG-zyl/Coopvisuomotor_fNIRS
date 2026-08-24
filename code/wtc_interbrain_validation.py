from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats
from scipy.io import loadmat


SCRIPT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("COOPVM_DATASET_ROOT", str(SCRIPT_ROOT.parent)))
PROCESSED_WTC_ROOT = Path(
    os.environ.get("COOPVM_PROCESSED_WTC_ROOT", str(DATASET_ROOT / "derivatives" / "homer3-wtc"))
)
BEH_ROOT = DATASET_ROOT
REPORT_ROOT = Path(
    os.environ.get(
        "COOPVM_REPORT_ROOT",
        str(DATASET_ROOT / "derivatives" / "technical-validation" / "reproduced"),
    )
)

RUN_CONDITIONS = {
    1: ("leader-follower", "easy", "carrot", "Leader-follower / carrot", "LF / carrot"),
    2: ("leader-follower", "hard", "wave", "Leader-follower / wave", "LF / wave"),
    3: ("egalitarian", "easy", "carrot", "Egalitarian / carrot", "EG / carrot"),
    4: ("egalitarian", "hard", "wave", "Egalitarian / wave", "EG / wave"),
}

EXPECTED_PARAMS = {
    "enPruneChannels_dRange": [0.01, 1750.0],
    "enPruneChannels_SNRthresh": 2.0,
    "enPruneChannels_SDrange": [0.0, 45.0],
    "enPruneChannels_reset": 1.0,
    "hmrMotionArtifactByChannel_tMotion": 0.5,
    "hmrMotionArtifactByChannel_tMask": 1.0,
    "hmrMotionArtifactByChannel_STDEVthresh": 20.0,
    "hmrMotionArtifactByChannel_AMPthresh": 0.5,
    "hmrMotionCorrectSpline_p": 0.99,
    "hmrMotionCorrectSpline_turnon": 1.0,
    "hmrOD2Conc_ppf": [6.0, 6.0, 6.0],
}

W0 = 6.0
TARGET_FS = 2.0
FOI_HZ = tuple(float(x) for x in os.environ.get("WTC_FOI", "0.03,0.33").split(","))
N_FREQUENCIES = 24
TASK_WINDOW_S = (5.0, 25.0)
BASELINE_WINDOW_S = (-15.0, 0.0)
USE_BASELINE_CORRECTION = False
CHANNEL_CLEAN_THRESHOLD = 0.80
TRIAL_CLEAN_THRESHOLD = 0.80
MIN_VALID_WINDOW_SECONDS = 30.0
PRIMARY_BEHAVIOR = "TotalScore"
BEHAVIOR_COLUMNS = ["TotalScore", "PositionScore", "ForceScore", "CompleteTime_s", "PerformanceEfficiency"]
CHANNEL_PAIR_MAP = [
    (1, 27),
    (2, 28),
    (3, 29),
    (4, 30),
    (5, 31),
    (6, 32),
    (7, 33),
    (8, 34),
    (9, 35),
    (10, 36),
    (11, 37),
    (12, 38),
    (13, 39),
    (14, 40),
    (15, 43),
    (16, 44),
    (17, 45),
    (18, 41),
    (19, 42),
    (20, 46),
    (21, 47),
    (22, 48),
    (23, 49),
    (24, 50),
    (25, 51),
    (26, 52),
]

WTC_METRIC_NAME = "baseline-corrected task-induced WTC" if USE_BASELINE_CORRECTION else "task-window WTC"
WTC_METRIC_SHORT = "Delta WTC" if USE_BASELINE_CORRECTION else "WTC"
WTC_OUTPUT_PREFIX = "wtc_interbrain_validation_baseline" if USE_BASELINE_CORRECTION else "wtc_interbrain_validation_taskwindow"


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_file_id(path: Path) -> tuple[int, int]:
    match = re.match(r"dyad-(\d+)_task-coopvm_run-(\d+)_nirs\.nirs$", path.name)
    if not match:
        raise ValueError(f"Unexpected NIRS filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def get_field(obj, name: str, default=None):
    return getattr(obj, name, default) if hasattr(obj, name) else default


def as_2d(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def as_float_array(value) -> np.ndarray:
    arr = np.asarray(value).astype(float).squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def values_match(actual, expected, tol=1e-8) -> bool:
    actual_arr = as_float_array(actual)
    expected_arr = np.asarray(expected, dtype=float).squeeze()
    if expected_arr.ndim == 0:
        expected_arr = expected_arr.reshape(1)
    return actual_arr.shape == expected_arr.shape and bool(np.all(np.abs(actual_arr - expected_arr) <= tol))


def value_to_string(value) -> str:
    arr = as_float_array(value)
    if arr.size == 1:
        return f"{float(arr[0]):.10g}"
    return " ".join(f"{float(x):.10g}" for x in arr.ravel())


def safe_fs(t: np.ndarray) -> float:
    t = np.ravel(np.asarray(t, dtype=float))
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def channel_map_from_ml(ml: np.ndarray, n_measurements: int) -> tuple[np.ndarray, list[str]]:
    ml = np.asarray(ml)
    if ml.ndim == 2 and ml.shape[0] == n_measurements and ml.shape[1] >= 2:
        pair_to_channel: dict[tuple[int, int], int] = {}
        channel_idx = np.zeros(n_measurements, dtype=int)
        labels: list[str] = []
        for i, row in enumerate(ml):
            pair = (int(row[0]), int(row[1]))
            if pair not in pair_to_channel:
                pair_to_channel[pair] = len(pair_to_channel)
                labels.append(f"S{pair[0]}-D{pair[1]}")
            channel_idx[i] = pair_to_channel[pair]
        return channel_idx, labels

    if n_measurements % 52 == 0:
        return np.repeat(np.arange(52), n_measurements // 52), [f"Ch{c:02d}" for c in range(1, 53)]
    return np.arange(n_measurements), [f"Meas{c:03d}" for c in range(1, n_measurements + 1)]


def group_measurements_to_channels(data: np.ndarray, channel_idx: np.ndarray, n_channels: int) -> np.ndarray:
    data = as_2d(data)
    out = np.full((data.shape[0], n_channels), np.nan)
    for ch in range(n_channels):
        cols = np.where(channel_idx == ch)[0]
        if cols.size:
            out[:, ch] = np.nanmean(data[:, cols], axis=1)
    return out


def channel_clean_matrix_from_tinc(t_inc_ch: np.ndarray, n_time: int, channel_idx: np.ndarray, n_channels: int) -> np.ndarray:
    arr = np.asarray(t_inc_ch, dtype=float)
    if arr.size == 0:
        return np.ones((n_time, n_channels), dtype=float)
    arr = as_2d(arr)
    if arr.shape[0] != n_time and arr.shape[1] == n_time:
        arr = arr.T
    if arr.shape[0] != n_time:
        return np.ones((n_time, n_channels), dtype=float)
    arr = (arr > 0).astype(float)
    if arr.shape[1] == n_channels:
        return arr
    if arr.shape[1] == len(channel_idx):
        return group_measurements_to_channels(arr, channel_idx, n_channels)
    return np.ones((n_time, n_channels), dtype=float)


def morlet_fourier_factor(w0: float = W0) -> float:
    return (4.0 * math.pi) / (w0 + math.sqrt(2.0 + w0 * w0))


def morlet_cwt_fft(x: np.ndarray, fs: float, frequencies: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    dt = 1.0 / fs
    finite = np.isfinite(x)
    if finite.mean() < 0.90:
        return np.full((frequencies.size, x.size), np.nan + 1j * np.nan, dtype=np.complex64)
    if not finite.all():
        idx = np.flatnonzero(finite)
        x = np.interp(np.arange(x.size), idx, x[finite])
    x = signal.detrend(x, type="linear")
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 0:
        return np.full((frequencies.size, x.size), np.nan + 1j * np.nan, dtype=np.complex64)
    x = (x - np.nanmean(x)) / sd

    fourier_factor = morlet_fourier_factor()
    out = np.empty((frequencies.size, x.size), dtype=np.complex64)
    for i, freq in enumerate(frequencies):
        period = 1.0 / freq
        scale = period / fourier_factor
        half_width = max(4, int(math.ceil(6.0 * scale * fs)))
        tt = np.arange(-half_width, half_width + 1, dtype=float) * dt
        eta = tt / scale
        wavelet = (math.pi ** -0.25) * np.exp(1j * W0 * eta) * np.exp(-(eta**2) / 2.0)
        kernel = np.conj(wavelet[::-1]) / math.sqrt(scale)
        out[i, :] = signal.fftconvolve(x, kernel, mode="same").astype(np.complex64) * dt
    return out


def smooth_complex(arr: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    real = ndimage.gaussian_filter(np.real(arr), sigma=sigma, mode="nearest")
    imag = ndimage.gaussian_filter(np.imag(arr), sigma=sigma, mode="nearest")
    return real + 1j * imag


def smooth_real(arr: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    return ndimage.gaussian_filter(np.asarray(arr, dtype=float), sigma=sigma, mode="nearest")


def wtc_from_cwt(wx: np.ndarray, wy: np.ndarray, fs: float, frequencies: np.ndarray) -> np.ndarray:
    n = min(wx.shape[1], wy.shape[1])
    wx = wx[:, :n].astype(np.complex64, copy=False)
    wy = wy[:, :n].astype(np.complex64, copy=False)
    scales = (1.0 / frequencies) / morlet_fourier_factor()
    inv_scale = (1.0 / scales)[:, None]
    sigma = (1.0, max(2.0, 2.5 * fs))
    swxy = smooth_complex(inv_scale * wx * np.conj(wy), sigma)
    swx = smooth_real(inv_scale * np.abs(wx) ** 2, sigma)
    swy = smooth_real(inv_scale * np.abs(wy) ** 2, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        wtc = (np.abs(swxy) ** 2) / (swx * swy)
    return np.clip(wtc, 0.0, 1.0)


def band_mean_wtc(
    wx: np.ndarray,
    wy: np.ndarray,
    fs: float,
    frequencies: np.ndarray,
    window_mask: np.ndarray,
) -> tuple[float, int]:
    n = min(wx.shape[1], wy.shape[1], window_mask.size)
    if n < 10:
        return np.nan, 0
    wx = wx[:, :n]
    wy = wy[:, :n]
    window_mask = window_mask[:n].astype(bool)
    if window_mask.mean() == 0:
        return np.nan, 0
    wtc = wtc_from_cwt(wx, wy, fs, frequencies)
    times = np.arange(n, dtype=float) / fs
    total_duration = n / fs
    fourier_factor = morlet_fourier_factor()
    valid = np.zeros_like(wtc, dtype=bool)
    for i, freq in enumerate(frequencies):
        scale = (1.0 / freq) / fourier_factor
        edge_margin_s = math.sqrt(2.0) * scale
        coi_ok = (times >= edge_margin_s) & (times <= total_duration - edge_margin_s)
        valid[i, :] = window_mask & coi_ok
    n_valid = int(np.sum(valid))
    if n_valid < int(MIN_VALID_WINDOW_SECONDS * fs):
        return np.nan, n_valid
    return float(np.nanmean(wtc[valid])), n_valid


def false_discovery_rate_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    p_f = p[finite]
    order = np.argsort(p_f)
    ranked = p_f[order]
    m = ranked.size
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    q_f = np.empty_like(adj)
    q_f[order] = adj
    q[finite] = q_f
    return q


def load_behavior_summary() -> pd.DataFrame:
    rows = []
    for dyad_dir in sorted(DATASET_ROOT.glob("dyad-*")):
        if not dyad_dir.is_dir():
            continue
        for beh_path in sorted((dyad_dir / "beh").glob("*_beh-trials.csv")):
            match = re.search(r"dyad-(\d+)_task-coopvm_run-(\d+)_beh-trials\.csv$", beh_path.name)
            if not match:
                continue
            dyad = int(match.group(1))
            run = int(match.group(2))
            df = pd.read_csv(beh_path)
            mode, difficulty, pattern, condition_label, condition_short = RUN_CONDITIONS[run]
            row = {
                "dyad": f"dyad-{dyad:03d}",
                "run": f"run-{run:02d}",
                "mode": mode,
                "difficulty": difficulty,
                "pattern": pattern,
                "condition_label": condition_label,
                "condition_short": condition_short,
                "n_behavior_trials": int(len(df)),
                "TotalScore": float(df["TotalScore"].mean()),
                "PositionScore": float(df["PositionScore"].mean()),
                "ForceScore": float(df["ForceScore"].mean()),
                "CompleteTime_s": float(df["CompleteTime_s"].mean()),
            }
            rows.append(row)
    beh = pd.DataFrame(rows)
    beh["PerformanceEfficiency"] = np.nan
    for run, idx in beh.groupby("run").groups.items():
        sub = beh.loc[idx]
        total_z = (sub["TotalScore"] - sub["TotalScore"].mean()) / sub["TotalScore"].std(ddof=1)
        time_z = (sub["CompleteTime_s"] - sub["CompleteTime_s"].mean()) / sub["CompleteTime_s"].std(ddof=1)
        beh.loc[idx, "PerformanceEfficiency"] = total_z - time_z
    return beh


def window_mask_from_trials(t_grid: np.ndarray, trial_starts: np.ndarray, window_s: tuple[float, float]) -> np.ndarray:
    mask = np.zeros(t_grid.size, dtype=bool)
    for t0 in trial_starts:
        mask |= (t_grid >= t0 + window_s[0]) & (t_grid < t0 + window_s[1])
    return mask


def load_run_data(proc_path: Path, frequencies: np.ndarray) -> dict:
    dyad, run = parse_file_id(proc_path)
    mat = loadmat(
        proc_path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["t", "s", "ml", "SD", "procInput", "procResult"],
    )
    t = np.ravel(np.asarray(mat["t"], dtype=float))
    s = as_2d(np.asarray(mat["s"], dtype=float))
    pr = mat["procResult"]
    dc = np.asarray(pr.dc, dtype=float)
    hbo = dc[:, 0, :] * 1e6
    n_time, n_channels = hbo.shape
    fs = safe_fs(t)
    t_grid = np.arange(t[0], t[-1], 1.0 / TARGET_FS)
    hbo_grid = np.full((t_grid.size, n_channels), np.nan, dtype=np.float32)
    for ch in range(n_channels):
        x = hbo[:, ch]
        finite = np.isfinite(x)
        if finite.mean() >= 0.90:
            hbo_grid[:, ch] = np.interp(t_grid, t[finite], x[finite]).astype(np.float32)

    ml = np.asarray(mat["ml"])
    channel_idx, channel_labels = channel_map_from_ml(ml, 156)
    t_inc_ch = np.asarray(get_field(pr, "tIncChAuto", []), dtype=float)
    clean_time = channel_clean_matrix_from_tinc(t_inc_ch, n_time, channel_idx, n_channels)
    channel_clean_fraction = np.nanmean(clean_time, axis=0)
    channel_clean = channel_clean_fraction >= CHANNEL_CLEAN_THRESHOLD
    trial_starts = t[np.flatnonzero(s[:, 0] > 0)]
    task_mask = window_mask_from_trials(t_grid, trial_starts, TASK_WINDOW_S)
    baseline_mask = window_mask_from_trials(t_grid, trial_starts, BASELINE_WINDOW_S)

    trial_clean_rows = []
    for i_trial, t0 in enumerate(trial_starts, 1):
        raw_mask = (t >= t0 + TASK_WINDOW_S[0]) & (t < t0 + TASK_WINDOW_S[1])
        if raw_mask.any():
            mean_clean = float(np.nanmean(clean_time[raw_mask, :]))
        else:
            mean_clean = np.nan
        trial_clean_rows.append((i_trial, mean_clean, bool(np.isfinite(mean_clean) and mean_clean >= TRIAL_CLEAN_THRESHOLD)))

    cwt = []
    for ch in range(n_channels):
        cwt.append(morlet_cwt_fft(hbo_grid[:, ch], TARGET_FS, frequencies))
    cwt_arr = np.stack(cwt, axis=0).astype(np.complex64)

    pp = mat["procInput"].procParam
    param_matches = {}
    for name, expected in EXPECTED_PARAMS.items():
        param_matches[name] = values_match(get_field(pp, name, np.nan), expected)
    bandpass_present = any("Bandpass" in a or "bandpass" in a for a in dir(pp) if not a.startswith("_"))

    return {
        "dyad_int": dyad,
        "dyad": f"dyad-{dyad:03d}",
        "run_int": run,
        "run": f"run-{run:02d}",
        "proc_path": str(proc_path),
        "raw_fs_hz": fs,
        "target_fs_hz": TARGET_FS,
        "n_time_original": n_time,
        "n_time_resampled": t_grid.size,
        "channel_labels": channel_labels,
        "channel_clean_fraction": channel_clean_fraction,
        "channel_clean": channel_clean,
        "task_mask": task_mask,
        "baseline_mask": baseline_mask,
        "trial_clean_rows": trial_clean_rows,
        "cwt": cwt_arr,
        "param_matches": param_matches,
        "bandpass_present": bandpass_present,
    }


def collect_wtc_metrics(out_dir: Path, frequencies: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proc_files = sorted(PROCESSED_WTC_ROOT.glob("dyad-*/*.nirs"))
    behavior = load_behavior_summary()
    behavior.to_csv(out_dir / "behavior_summary_by_dyad_run.csv", index=False, encoding="utf-8-sig")

    param_rows = []
    trial_qc_rows = []
    metric_rows = []

    for run in [1, 2, 3, 4]:
        files = [p for p in proc_files if parse_file_id(p)[1] == run]
        run_data = []
        print(f"Loading/CWT run-{run:02d}: {len(files)} files", flush=True)
        for i, proc_path in enumerate(files, 1):
            data = load_run_data(proc_path, frequencies)
            run_data.append(data)
            mode, difficulty, pattern, condition_label, condition_short = RUN_CONDITIONS[run]
            for pname, ok in data["param_matches"].items():
                param_rows.append(
                    {
                        "dyad": data["dyad"],
                        "run": data["run"],
                        "parameter": pname,
                        "matches_expected": bool(ok),
                    }
                )
            param_rows.append(
                {
                    "dyad": data["dyad"],
                    "run": data["run"],
                    "parameter": "hmrBandpassFilt_absent",
                    "matches_expected": bool(not data["bandpass_present"]),
                }
            )
            for trial_idx, clean_fraction, retained in data["trial_clean_rows"]:
                trial_qc_rows.append(
                    {
                        "dyad": data["dyad"],
                        "run": data["run"],
                        "condition_label": condition_label,
                        "trial_index": trial_idx,
                        "trial_clean_fraction": clean_fraction,
                        "retained_for_wtc": retained,
                    }
                )
            if i % 8 == 0 or i == len(files):
                print(f"  CWT ready: {i}/{len(files)}", flush=True)

        run_data = sorted(run_data, key=lambda x: x["dyad_int"])
        dyads = [d["dyad"] for d in run_data]
        mode, difficulty, pattern, condition_label, condition_short = RUN_CONDITIONS[run]
        beh_run = behavior[behavior["run"].eq(f"run-{run:02d}")].set_index("dyad")
        n_dyads = len(run_data)

        for i, data in enumerate(run_data):
            pseudo = run_data[(i + 1) % n_dyads]
            beh = beh_run.loc[data["dyad"]].to_dict()
            for pair_idx, (ch_a_global, ch_b_global) in enumerate(CHANNEL_PAIR_MAP, 1):
                ch_a = ch_a_global - 1
                ch_b = ch_b_global - 1
                clean_a = bool(data["channel_clean"][ch_a])
                clean_b = bool(data["channel_clean"][ch_b])
                clean_pseudo_b = bool(pseudo["channel_clean"][ch_b])
                true_task_wtc, true_task_valid = (np.nan, 0)
                true_baseline_wtc, true_baseline_valid = (np.nan, 0)
                pseudo_task_wtc, pseudo_task_valid = (np.nan, 0)
                pseudo_baseline_wtc, pseudo_baseline_valid = (np.nan, 0)
                if clean_a and clean_b:
                    true_task_wtc, true_task_valid = band_mean_wtc(
                        data["cwt"][ch_a],
                        data["cwt"][ch_b],
                        TARGET_FS,
                        frequencies,
                        data["task_mask"],
                    )
                    true_baseline_wtc, true_baseline_valid = band_mean_wtc(
                        data["cwt"][ch_a],
                        data["cwt"][ch_b],
                        TARGET_FS,
                        frequencies,
                        data["baseline_mask"],
                    )
                if clean_a and clean_pseudo_b:
                    pseudo_task_wtc, pseudo_task_valid = band_mean_wtc(
                        data["cwt"][ch_a],
                        pseudo["cwt"][ch_b],
                        TARGET_FS,
                        frequencies,
                        data["task_mask"],
                    )
                    pseudo_baseline_wtc, pseudo_baseline_valid = band_mean_wtc(
                        data["cwt"][ch_a],
                        pseudo["cwt"][ch_b],
                        TARGET_FS,
                        frequencies,
                        data["baseline_mask"],
                    )
                if USE_BASELINE_CORRECTION:
                    true_wtc = (
                        true_task_wtc - true_baseline_wtc
                        if np.isfinite(true_task_wtc) and np.isfinite(true_baseline_wtc)
                        else np.nan
                    )
                    pseudo_wtc = (
                        pseudo_task_wtc - pseudo_baseline_wtc
                        if np.isfinite(pseudo_task_wtc) and np.isfinite(pseudo_baseline_wtc)
                        else np.nan
                    )
                else:
                    true_wtc = true_task_wtc
                    pseudo_wtc = pseudo_task_wtc
                true_delta_wtc = (
                    true_task_wtc - true_baseline_wtc
                    if np.isfinite(true_task_wtc) and np.isfinite(true_baseline_wtc)
                    else np.nan
                )
                pseudo_delta_wtc = (
                    pseudo_task_wtc - pseudo_baseline_wtc
                    if np.isfinite(pseudo_task_wtc) and np.isfinite(pseudo_baseline_wtc)
                    else np.nan
                )
                row = {
                    "dyad": data["dyad"],
                    "pseudo_partner": pseudo["dyad"],
                    "run": data["run"],
                    "mode": mode,
                    "difficulty": difficulty,
                    "pattern": pattern,
                    "condition_label": condition_label,
                    "condition_short": condition_short,
                    "channel_pair": pair_idx,
                    "channel_A_global": ch_a_global,
                    "channel_B_global": ch_b_global,
                    "channel_A_label": data["channel_labels"][ch_a],
                    "channel_B_label": data["channel_labels"][ch_b],
                    "true_wtc": true_wtc,
                    "pseudo_wtc": pseudo_wtc,
                    "true_task_wtc": true_task_wtc,
                    "true_baseline_wtc": true_baseline_wtc,
                    "true_delta_wtc": true_delta_wtc,
                    "pseudo_task_wtc": pseudo_task_wtc,
                    "pseudo_baseline_wtc": pseudo_baseline_wtc,
                    "pseudo_delta_wtc": pseudo_delta_wtc,
                    "true_minus_pseudo": true_wtc - pseudo_wtc
                    if np.isfinite(true_wtc) and np.isfinite(pseudo_wtc)
                    else np.nan,
                    "n_valid_true_task_points": true_task_valid,
                    "n_valid_true_baseline_points": true_baseline_valid,
                    "n_valid_pseudo_task_points": pseudo_task_valid,
                    "n_valid_pseudo_baseline_points": pseudo_baseline_valid,
                    "channel_A_clean_fraction": float(data["channel_clean_fraction"][ch_a]),
                    "channel_B_clean_fraction": float(data["channel_clean_fraction"][ch_b]),
                    "channel_A_retained": clean_a,
                    "channel_B_retained": clean_b,
                    "pseudo_channel_B_retained": clean_pseudo_b,
                }
                for col in BEHAVIOR_COLUMNS:
                    row[col] = beh[col]
                metric_rows.append(row)
        print(f"WTC metrics done for run-{run:02d}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    param_df = pd.DataFrame(param_rows)
    trial_qc = pd.DataFrame(trial_qc_rows)
    metrics.to_csv(out_dir / "wtc_metrics_by_dyad_run_channel.csv", index=False, encoding="utf-8-sig")
    param_df.to_csv(out_dir / "wtc_preprocessing_parameter_check.csv", index=False, encoding="utf-8-sig")
    trial_qc.to_csv(out_dir / "wtc_trial_qc_summary.csv", index=False, encoding="utf-8-sig")
    param_df.loc[~param_df["matches_expected"].astype(bool)].to_csv(
        out_dir / "wtc_preprocessing_parameter_mismatches.csv", index=False, encoding="utf-8-sig"
    )
    return metrics, behavior, param_df, trial_qc


def summarize_wtc(metrics: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for (run, condition_label, condition_short, ch), sub in metrics.groupby(
        ["run", "condition_label", "condition_short", "channel_pair"]
    ):
        true = sub["true_wtc"].to_numpy(float)
        pseudo = sub["pseudo_wtc"].to_numpy(float)
        paired = sub[["true_wtc", "pseudo_wtc"]].dropna()
        if len(paired) >= 8:
            try:
                wil = stats.wilcoxon(
                    paired["true_wtc"].to_numpy(float),
                    paired["pseudo_wtc"].to_numpy(float),
                    alternative="greater",
                    zero_method="wilcox",
                )
                p_val = float(wil.pvalue)
            except Exception:
                p_val = np.nan
            dz = float((paired["true_wtc"] - paired["pseudo_wtc"]).mean() / (paired["true_wtc"] - paired["pseudo_wtc"]).std(ddof=1))
        else:
            p_val = np.nan
            dz = np.nan
        rows.append(
            {
                "run": run,
                "condition_label": condition_label,
                "condition_short": condition_short,
                "channel_pair": int(ch),
                "n_true": int(np.sum(np.isfinite(true))),
                "n_paired": int(len(paired)),
                "true_mean_wtc": float(np.nanmean(true)),
                "true_sd_wtc": float(np.nanstd(true, ddof=1)),
                "pseudo_mean_wtc": float(np.nanmean(pseudo)),
                "pseudo_sd_wtc": float(np.nanstd(pseudo, ddof=1)),
                "true_minus_pseudo_mean": float(np.nanmean(true - pseudo)),
                "wilcoxon_true_greater_p": p_val,
                "paired_effect_dz": dz,
            }
        )
    group = pd.DataFrame(rows)
    group["wilcoxon_true_greater_q_fdr"] = false_discovery_rate_bh(group["wilcoxon_true_greater_p"].to_numpy(float))
    group.to_csv(out_dir / "wtc_group_summary_by_condition_channel.csv", index=False, encoding="utf-8-sig")

    condition = (
        metrics.groupby(["run", "condition_label", "condition_short"], as_index=False)
        .agg(
            n_observations=("true_wtc", lambda x: int(np.sum(np.isfinite(x)))),
            true_mean_wtc=("true_wtc", "mean"),
            true_sd_wtc=("true_wtc", "std"),
            pseudo_mean_wtc=("pseudo_wtc", "mean"),
            pseudo_sd_wtc=("pseudo_wtc", "std"),
            true_minus_pseudo_mean=("true_minus_pseudo", "mean"),
        )
        .sort_values("run")
    )
    condition.to_csv(out_dir / "wtc_group_summary_by_condition.csv", index=False, encoding="utf-8-sig")

    dyad_means = (
        metrics.groupby(["run", "condition_label", "condition_short", "dyad"], as_index=False)
        .agg(true_wtc=("true_wtc", "mean"), pseudo_wtc=("pseudo_wtc", "mean"))
        .sort_values(["run", "dyad"])
    )
    comparison_rows = []
    for (run, condition_label, condition_short), sub in dyad_means.groupby(
        ["run", "condition_label", "condition_short"]
    ):
        paired = sub[["true_wtc", "pseudo_wtc"]].dropna()
        diff = paired["true_wtc"] - paired["pseudo_wtc"]
        test = stats.ttest_rel(paired["true_wtc"], paired["pseudo_wtc"])
        comparison_rows.append(
            {
                "run": run,
                "condition_label": condition_label,
                "condition_short": condition_short,
                "n_dyads": int(len(paired)),
                "true_dyadmean_wtc": float(paired["true_wtc"].mean()),
                "true_dyadmean_sd": float(paired["true_wtc"].std(ddof=1)),
                "pseudo_dyadmean_wtc": float(paired["pseudo_wtc"].mean()),
                "pseudo_dyadmean_sd": float(paired["pseudo_wtc"].std(ddof=1)),
                "true_minus_pseudo_mean": float(diff.mean()),
                "true_minus_pseudo_sd": float(diff.std(ddof=1)),
                "paired_t": float(test.statistic),
                "df": int(len(paired) - 1),
                "p_two_sided": float(test.pvalue),
                "cohen_dz": float(diff.mean() / diff.std(ddof=1)),
            }
        )
    pd.DataFrame(comparison_rows).sort_values("run").to_csv(
        out_dir / "table_wtc_true_vs_pseudo_dyadmean_by_condition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corr_rows = []
    for behavior_col in BEHAVIOR_COLUMNS:
        for (run, condition_label, condition_short, ch), sub in metrics.groupby(
            ["run", "condition_label", "condition_short", "channel_pair"]
        ):
            use = sub[["true_wtc", behavior_col, "dyad"]].dropna()
            if len(use) >= 8 and use["true_wtc"].nunique() > 2 and use[behavior_col].nunique() > 2:
                rho, p = stats.spearmanr(use["true_wtc"], use[behavior_col])
                rho, p = float(rho), float(p)
            else:
                rho, p = np.nan, np.nan
            corr_rows.append(
                {
                    "behavior_metric": behavior_col,
                    "run": run,
                    "condition_label": condition_label,
                    "condition_short": condition_short,
                    "channel_pair": int(ch),
                    "n_dyads": int(len(use)),
                    "spearman_rho": rho,
                    "p_uncorrected": p,
                }
            )
    corr = pd.DataFrame(corr_rows)
    corr["q_fdr_within_metric"] = np.nan
    for metric, idx in corr.groupby("behavior_metric").groups.items():
        corr.loc[idx, "q_fdr_within_metric"] = false_discovery_rate_bh(corr.loc[idx, "p_uncorrected"].to_numpy(float))
    corr.to_csv(out_dir / "wtc_behavior_correlations_all_metrics.csv", index=False, encoding="utf-8-sig")
    primary = corr[corr["behavior_metric"].eq(PRIMARY_BEHAVIOR)].copy()
    primary.to_csv(out_dir / "wtc_behavior_correlations_primary_TotalScore.csv", index=False, encoding="utf-8-sig")
    return group, condition, corr


def save_table_image(df: pd.DataFrame, out_png: Path, out_pdf: Path, title: str, col_widths: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 0.50 * (len(df) + 1) + 0.9))
    ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.36)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#F1F3F5")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#FAFAFA")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_figures(metrics: pd.DataFrame, group: pd.DataFrame, condition: pd.DataFrame, corr: pd.DataFrame, out_dir: Path):
    condition_order = [RUN_CONDITIONS[i][3] for i in [1, 2, 3, 4]]
    condition_short = [RUN_CONDITIONS[i][4] for i in [1, 2, 3, 4]]
    chs = np.arange(1, 27)

    true_mat = np.full((26, 4), np.nan)
    diff_mat = np.full((26, 4), np.nan)
    rho_mat = np.full((26, 4), np.nan)
    q_mat = np.full((26, 4), np.nan)
    for j, run in enumerate([f"run-{i:02d}" for i in [1, 2, 3, 4]]):
        for ch in chs:
            sub = group[(group["run"].eq(run)) & (group["channel_pair"].eq(ch))]
            if len(sub):
                true_mat[ch - 1, j] = float(sub["true_mean_wtc"].iloc[0])
                diff_mat[ch - 1, j] = float(sub["true_minus_pseudo_mean"].iloc[0])
            csub = corr[
                corr["behavior_metric"].eq(PRIMARY_BEHAVIOR)
                & corr["run"].eq(run)
                & corr["channel_pair"].eq(ch)
            ]
            if len(csub):
                rho_mat[ch - 1, j] = float(csub["spearman_rho"].iloc[0])
                q_mat[ch - 1, j] = float(csub["q_fdr_within_metric"].iloc[0])

    primary = corr[corr["behavior_metric"].eq(PRIMARY_BEHAVIOR)].copy()
    sig_pos = primary[(primary["spearman_rho"] > 0) & (primary["q_fdr_within_metric"] < 0.05)].sort_values(
        ["q_fdr_within_metric", "p_uncorrected", "spearman_rho"], ascending=[True, True, False]
    )
    if len(sig_pos):
        scatter_pick = sig_pos.iloc[0]
        scatter_note = "FDR significant"
    else:
        scatter_pick = primary[primary["spearman_rho"] > 0].sort_values(
            ["p_uncorrected", "spearman_rho"], ascending=[True, False]
        ).iloc[0]
        scatter_note = "strongest positive uncorrected"

    fig = plt.figure(figsize=(10.2, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    im = ax_a.imshow(true_mat, aspect="auto", origin="lower", cmap="viridis", vmin=np.nanpercentile(true_mat, 5), vmax=np.nanpercentile(true_mat, 95))
    ax_a.set_title(f"A. {WTC_METRIC_NAME.capitalize()}")
    ax_a.set_xticks(np.arange(4))
    ax_a.set_xticklabels(condition_short, rotation=25, ha="right")
    ax_a.set_yticks(np.arange(0, 26, 5))
    ax_a.set_yticklabels([str(i) for i in range(1, 27, 5)])
    ax_a.set_xlabel("Condition")
    ax_a.set_ylabel("Corresponding channel pair")
    cb = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.03)
    cb.set_label(WTC_METRIC_SHORT)

    ax_b = fig.add_subplot(gs[0, 1])
    plot_df = metrics.dropna(subset=["true_wtc", "pseudo_wtc"]).copy()
    conds = [f"run-{i:02d}" for i in [1, 2, 3, 4]]
    positions = np.arange(4)
    true_vals = [plot_df[plot_df["run"].eq(run)]["true_wtc"].to_numpy(float) for run in conds]
    pseudo_vals = [plot_df[plot_df["run"].eq(run)]["pseudo_wtc"].to_numpy(float) for run in conds]
    bp1 = ax_b.boxplot(true_vals, positions=positions - 0.16, widths=0.24, patch_artist=True, showfliers=False)
    bp2 = ax_b.boxplot(pseudo_vals, positions=positions + 0.16, widths=0.24, patch_artist=True, showfliers=False)
    for p in bp1["boxes"]:
        p.set_facecolor("#4C78A8")
        p.set_alpha(0.70)
    for p in bp2["boxes"]:
        p.set_facecolor("#F28E2B")
        p.set_alpha(0.70)
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(condition_short, rotation=25, ha="right")
    ax_b.set_ylabel(WTC_METRIC_SHORT)
    ax_b.set_title("B. True dyads vs pseudo dyads")
    ax_b.legend([bp1["boxes"][0], bp2["boxes"][0]], ["True", "Pseudo"], frameon=False, loc="best")
    ax_b.grid(axis="y", alpha=0.16)

    ax_c = fig.add_subplot(gs[1, 0])
    max_abs = np.nanmax(np.abs(rho_mat))
    im2 = ax_c.imshow(rho_mat, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    for row in range(26):
        for col in range(4):
            if np.isfinite(q_mat[row, col]) and q_mat[row, col] < 0.05:
                ax_c.text(col, row, "*", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
    ax_c.set_title(f"C. {WTC_METRIC_SHORT}-{PRIMARY_BEHAVIOR} association")
    ax_c.set_xticks(np.arange(4))
    ax_c.set_xticklabels(condition_short, rotation=25, ha="right")
    ax_c.set_yticks(np.arange(0, 26, 5))
    ax_c.set_yticklabels([str(i) for i in range(1, 27, 5)])
    ax_c.set_xlabel("Condition")
    ax_c.set_ylabel("Corresponding channel pair")
    cb2 = fig.colorbar(im2, ax=ax_c, fraction=0.046, pad=0.03)
    cb2.set_label("Spearman rho")

    ax_d = fig.add_subplot(gs[1, 1])
    run = scatter_pick["run"]
    ch = int(scatter_pick["channel_pair"])
    sub = metrics[(metrics["run"].eq(run)) & (metrics["channel_pair"].eq(ch))].dropna(subset=["true_wtc", PRIMARY_BEHAVIOR])
    x = sub["true_wtc"].to_numpy(float)
    y = sub[PRIMARY_BEHAVIOR].to_numpy(float)
    ax_d.scatter(x, y, s=32, color="#333333", alpha=0.82, linewidths=0)
    if len(sub) >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax_d.plot(xs, slope * xs + intercept, color="#C0392B", lw=1.5)
    ax_d.set_xlabel(WTC_METRIC_SHORT)
    ax_d.set_ylabel(PRIMARY_BEHAVIOR)
    ax_d.set_title(
        f"D. {RUN_CONDITIONS[int(run[-2:])][4]}, Ch pair {ch:02d}\n"
        f"rho={scatter_pick['spearman_rho']:.2f}, p={scatter_pick['p_uncorrected']:.3f}, "
        f"q={scatter_pick['q_fdr_within_metric']:.3f}"
    )
    ax_d.grid(axis="y", alpha=0.16)

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Inter-brain synchronization validation using WTC", x=0.02, ha="left", fontweight="bold")
    fig.savefig(out_dir / "figure_wtc_interbrain_validation_fdr.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "figure_wtc_interbrain_validation_fdr.pdf", bbox_inches="tight")
    plt.close(fig)

    top_primary = primary.sort_values(["q_fdr_within_metric", "p_uncorrected", "spearman_rho"], ascending=[True, True, False]).head(12)
    top_primary_display = top_primary[
        ["condition_short", "channel_pair", "n_dyads", "spearman_rho", "p_uncorrected", "q_fdr_within_metric"]
    ].rename(
        columns={
            "condition_short": "Condition",
            "channel_pair": "Ch pair",
            "n_dyads": "N",
            "spearman_rho": "rho",
            "p_uncorrected": "p",
            "q_fdr_within_metric": "q FDR",
        }
    )
    for c in ["rho", "p", "q FDR"]:
        top_primary_display[c] = top_primary_display[c].map(lambda x: f"{float(x):.3f}")
    top_primary_display.to_csv(out_dir / "table_top_wtc_TotalScore_correlations.csv", index=False, encoding="utf-8-sig")
    save_table_image(
        top_primary_display,
        out_dir / "table_top_wtc_TotalScore_correlations.png",
        out_dir / "table_top_wtc_TotalScore_correlations.pdf",
        f"Top {WTC_METRIC_SHORT}-TotalScore correlations",
        [0.20, 0.12, 0.10, 0.14, 0.14, 0.14],
    )

    cond_display = condition.copy()
    for c in ["true_mean_wtc", "true_sd_wtc", "pseudo_mean_wtc", "pseudo_sd_wtc", "true_minus_pseudo_mean"]:
        cond_display[c] = cond_display[c].map(lambda x: f"{float(x):.3f}")
    cond_display = cond_display[
        ["condition_short", "n_observations", "true_mean_wtc", "true_sd_wtc", "pseudo_mean_wtc", "pseudo_sd_wtc", "true_minus_pseudo_mean"]
    ].rename(
        columns={
            "condition_short": "Condition",
            "n_observations": "N obs.",
            "true_mean_wtc": f"True {WTC_METRIC_SHORT}",
            "true_sd_wtc": "True SD",
            "pseudo_mean_wtc": f"Pseudo {WTC_METRIC_SHORT}",
            "pseudo_sd_wtc": "Pseudo SD",
            "true_minus_pseudo_mean": "True-pseudo",
        }
    )
    cond_display.to_csv(out_dir / "table_wtc_true_vs_pseudo_by_condition.csv", index=False, encoding="utf-8-sig")
    save_table_image(
        cond_display,
        out_dir / "table_wtc_true_vs_pseudo_by_condition.png",
        out_dir / "table_wtc_true_vs_pseudo_by_condition.pdf",
        f"{WTC_METRIC_NAME.capitalize()} summary by condition",
        [0.18, 0.10, 0.13, 0.12, 0.14, 0.13, 0.14],
    )

    return scatter_pick

def main():
    out_dir = REPORT_ROOT / f"{WTC_OUTPUT_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frequencies = np.geomspace(FOI_HZ[0], FOI_HZ[1], N_FREQUENCIES)
    pd.DataFrame(
        [
            {
                "channel_pair": i,
                "channel_A_global": a,
                "channel_B_global": b,
            }
            for i, (a, b) in enumerate(CHANNEL_PAIR_MAP, 1)
        ]
    ).to_csv(out_dir / "channel_pair_mapping.csv", index=False, encoding="utf-8-sig")
    
    metrics, behavior, param_df, trial_qc = collect_wtc_metrics(out_dir, frequencies)
    group, condition, corr = summarize_wtc(metrics, out_dir)
    scatter_pick = make_figures(metrics, group, condition, corr, out_dir)
    print("Analysis complete.", flush=True)


if __name__ == "__main__":
    main()
