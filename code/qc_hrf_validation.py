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
from scipy.io import loadmat


SCRIPT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("COOPVM_DATASET_ROOT", str(SCRIPT_ROOT.parent)))
SOURCEDATA_ROOT = Path(os.environ.get("COOPVM_SOURCEDATA_ROOT", str(DATASET_ROOT / "sourcedata")))
PROCESSED_ROOT = Path(
    os.environ.get("COOPVM_PROCESSED_ROOT", str(DATASET_ROOT / "derivatives" / "homer3-hrf"))
)
REPORT_ROOT = Path(
    os.environ.get(
        "COOPVM_REPORT_ROOT",
        str(DATASET_ROOT / "derivatives" / "technical-validation" / "reproduced"),
    )
)

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
    "hmrBandpassFilt_hpf": 0.01,
    "hmrBandpassFilt_lpf": 0.10,
    "hmrOD2Conc_ppf": [6.0, 6.0, 6.0],
    "hmrBlockAvg_trange": [-5.0, 35.0],
}

PARAM_TOL = 1e-8
CHANNEL_MOTION_CLEAN_THRESHOLD = 0.80
TRIAL_CLEAN_THRESHOLD = 0.80
RAW_CV_FLAG_THRESHOLD_PERCENT = 15.0
TRIAL_DURATION_SECONDS = 20.0
EPOCH_START_S = -5.0
EPOCH_END_S = 35.0
BASELINE_START_S = -5.0
BASELINE_END_S = 0.0
TARGET_FS = 11.0

RUN_CONDITIONS = {
    1: ("leader-follower", "easy", "carrot", "Leader-follower / carrot", "LF / carrot"),
    2: ("leader-follower", "hard", "wave", "Leader-follower / wave", "LF / wave"),
    3: ("egalitarian", "easy", "carrot", "Egalitarian / carrot", "EG / carrot"),
    4: ("egalitarian", "hard", "wave", "Egalitarian / wave", "EG / wave"),
}


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


def values_match(actual, expected, tol: float = PARAM_TOL) -> bool:
    actual_arr = as_float_array(actual)
    expected_arr = np.asarray(expected, dtype=float).squeeze()
    if expected_arr.ndim == 0:
        expected_arr = expected_arr.reshape(1)
    if actual_arr.shape != expected_arr.shape:
        return False
    return bool(np.all(np.isfinite(actual_arr)) and np.all(np.abs(actual_arr - expected_arr) <= tol))


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
        reps = n_measurements // 52
        return np.repeat(np.arange(52), reps), [f"Ch{c:02d}" for c in range(1, 53)]

    return np.arange(n_measurements), [f"Meas{c:03d}" for c in range(1, n_measurements + 1)]


def group_measurements_to_channels(data: np.ndarray, channel_idx: np.ndarray, n_channels: int) -> np.ndarray:
    data = as_2d(data)
    out = np.full((data.shape[0], n_channels), np.nan)
    for ch in range(n_channels):
        cols = np.where(channel_idx == ch)[0]
        if cols.size:
            out[:, ch] = np.nanmean(data[:, cols], axis=1)
    return out


def channel_active_from_measlistact(sd_obj, channel_idx: np.ndarray, n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    meas_act = get_field(sd_obj, "MeasListAct", None) if sd_obj is not None else None
    if meas_act is None:
        return np.ones(n_channels, dtype=bool), np.ones(n_channels, dtype=float)
    meas_act = np.ravel(np.asarray(meas_act, dtype=float))
    if meas_act.size == n_channels:
        active_fraction = (meas_act > 0).astype(float)
        return active_fraction > 0, active_fraction
    if meas_act.size == len(channel_idx):
        active_fraction = np.full(n_channels, np.nan)
        for ch in range(n_channels):
            cols = np.where(channel_idx == ch)[0]
            if cols.size:
                active_fraction[ch] = float(np.mean(meas_act[cols] > 0))
        return active_fraction >= 1.0, active_fraction
    return np.ones(n_channels, dtype=bool), np.ones(n_channels, dtype=float)


def channel_clean_matrix_from_tinc(t_inc_ch: np.ndarray, n_time: int, channel_idx: np.ndarray, n_channels: int) -> np.ndarray:
    if t_inc_ch is None:
        return np.ones((n_time, n_channels), dtype=float)
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


def raw_channel_cv_percent(d: np.ndarray, channel_idx: np.ndarray, n_channels: int) -> np.ndarray:
    d = as_2d(np.asarray(d, dtype=float))
    meas_mean = np.nanmean(d, axis=0)
    meas_std = np.nanstd(d, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        meas_cv = 100.0 * meas_std / np.abs(meas_mean)
    meas_cv[~np.isfinite(meas_cv)] = np.nan

    out = np.full(n_channels, np.nan)
    for ch in range(n_channels):
        cols = np.where(channel_idx == ch)[0]
        if cols.size:
            out[ch] = np.nanmedian(meas_cv[cols])
    return out


def trial_windows(t: np.ndarray, s: np.ndarray) -> list[tuple[int, float, np.ndarray]]:
    s = as_2d(s)
    starts = np.flatnonzero(s[:, 0] > 0)
    out = []
    fs = safe_fs(t)
    for idx0 in starts:
        t0 = float(t[idx0]) if idx0 < t.size else float(idx0 / fs)
        mask = (t >= t0) & (t < t0 + TRIAL_DURATION_SECONDS)
        out.append((int(idx0), t0, mask))
    return out


def load_proc(proc_path: Path):
    return loadmat(
        proc_path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["t", "s", "SD", "procInput", "procResult"],
    )


def collect_qc(out_dir: Path):
    raw_files = sorted(SOURCEDATA_ROOT.glob("dyad-*/*.nirs"))
    print(f"Found raw NIRS files: {len(raw_files)}", flush=True)

    time_rows = []
    param_rows = []
    run_rows = []
    channel_rows = []
    trial_rows = []
    hrf_inputs = []

    for i_file, raw_path in enumerate(raw_files, 1):
        dyad, run = parse_file_id(raw_path)
        proc_path = PROCESSED_ROOT / raw_path.parent.name / raw_path.name
        mode, difficulty, pattern, condition_label, condition_short = RUN_CONDITIONS[run]
        base = {
            "dyad": f"dyad-{dyad:03d}",
            "run": f"run-{run:02d}",
            "mode": mode,
            "difficulty": difficulty,
            "pattern": pattern,
            "condition_label": condition_label,
        }

        time_row = dict(base)
        time_row.update(
            {
                "processed_exists": proc_path.exists(),
                "raw_d_timepoints": np.nan,
                "raw_t_timepoints": np.nan,
                "raw_s_timepoints": np.nan,
                "processed_t_timepoints": np.nan,
                "processed_s_timepoints": np.nan,
                "processed_dc_timepoints": np.nan,
                "processed_tIncAuto_timepoints": np.nan,
                "processed_tIncChAuto_timepoints": np.nan,
                "timepoints_match": False,
                "reason": "",
            }
        )
        if not proc_path.exists():
            time_row["reason"] = "missing_processed_file"
            time_rows.append(time_row)
            continue

        try:
            raw = loadmat(raw_path, squeeze_me=True, struct_as_record=False, variable_names=["d", "t", "s", "ml"])
            proc = load_proc(proc_path)
            d = as_2d(raw["d"])
            raw_t = np.ravel(np.asarray(raw["t"], dtype=float))
            raw_s = as_2d(np.asarray(raw["s"], dtype=float))
            proc_t = np.ravel(np.asarray(proc.get("t", raw_t), dtype=float))
            proc_s = as_2d(np.asarray(proc.get("s", raw_s), dtype=float))
            pr = proc["procResult"]
            dc = np.asarray(get_field(pr, "dc", np.empty((0, 0, 0))), dtype=float)
            t_inc_auto = np.ravel(np.asarray(get_field(pr, "tIncAuto", []), dtype=float))
            t_inc_ch = np.asarray(get_field(pr, "tIncChAuto", []), dtype=float)
            n_time = d.shape[0]

            time_row.update(
                {
                    "raw_d_timepoints": n_time,
                    "raw_t_timepoints": raw_t.size,
                    "raw_s_timepoints": raw_s.shape[0],
                    "processed_t_timepoints": proc_t.size,
                    "processed_s_timepoints": proc_s.shape[0],
                    "processed_dc_timepoints": dc.shape[0] if dc.ndim == 3 else np.nan,
                    "processed_tIncAuto_timepoints": t_inc_auto.size if t_inc_auto.size else np.nan,
                    "processed_tIncChAuto_timepoints": t_inc_ch.shape[0] if t_inc_ch.size else np.nan,
                }
            )
            checks = [
                n_time == raw_t.size,
                n_time == raw_s.shape[0],
                n_time == proc_t.size,
                n_time == proc_s.shape[0],
                dc.ndim == 3 and n_time == dc.shape[0] and dc.shape[1] >= 2,
                t_inc_auto.size == 0 or n_time == t_inc_auto.size,
                t_inc_ch.size == 0 or n_time == t_inc_ch.shape[0],
            ]
            time_row["timepoints_match"] = bool(all(checks))
            time_row["reason"] = "ok" if time_row["timepoints_match"] else "timepoint_or_structure_mismatch"

            pp = get_field(proc.get("procInput", None), "procParam", None)
            for pname, expected in EXPECTED_PARAMS.items():
                actual = get_field(pp, pname, np.nan) if pp is not None else np.nan
                prow = dict(base)
                prow.update(
                    {
                        "parameter": pname,
                        "expected": value_to_string(expected),
                        "actual": value_to_string(actual),
                        "matches_expected": values_match(actual, expected),
                    }
                )
                param_rows.append(prow)

            if not time_row["timepoints_match"]:
                time_rows.append(time_row)
                continue

            ml = np.asarray(raw["ml"])
            channel_idx, channel_labels = channel_map_from_ml(ml, d.shape[1])
            n_channels = int(dc.shape[2])
            if len(channel_labels) != n_channels:
                channel_labels = [f"Ch{c:02d}" for c in range(1, n_channels + 1)]
                if d.shape[1] % n_channels == 0:
                    channel_idx = np.repeat(np.arange(n_channels), d.shape[1] // n_channels)
                else:
                    channel_idx = np.arange(d.shape[1])

            sd = proc.get("SD", get_field(pr, "SD", None))
            active_channels, active_fraction = channel_active_from_measlistact(sd, channel_idx, n_channels)
            clean_time = channel_clean_matrix_from_tinc(t_inc_ch, n_time, channel_idx, n_channels)
            motion_clean_fraction = np.nanmean(clean_time, axis=0)
            raw_cv = raw_channel_cv_percent(d, channel_idx, n_channels)
            motion_flagged = motion_clean_fraction < CHANNEL_MOTION_CLEAN_THRESHOLD
            enprune_flagged = ~active_channels
            final_excluded = enprune_flagged | motion_flagged
            retained = ~final_excluded

            windows = trial_windows(raw_t, raw_s)
            active_for_trials = active_channels if np.any(active_channels) else np.ones(n_channels, dtype=bool)
            trial_clean_values = []
            trial_excluded_values = []
            for i_trial, (idx0, t0, mask) in enumerate(windows, 1):
                if mask.any():
                    clean_fraction = float(np.nanmean(clean_time[mask, :][:, active_for_trials]))
                else:
                    clean_fraction = np.nan
                excluded_trial = bool(not np.isfinite(clean_fraction) or clean_fraction < TRIAL_CLEAN_THRESHOLD)
                trial_clean_values.append(clean_fraction)
                trial_excluded_values.append(excluded_trial)
                row = dict(base)
                row.update(
                    {
                        "trial_index": i_trial,
                        "start_sample": idx0,
                        "start_time_s": t0,
                        "clean_fraction_active_channels": clean_fraction,
                        "excluded_trial": excluded_trial,
                        "clean_fraction_source": "mean tIncChAuto across enPrune-retained channels within 20 s trial window",
                    }
                )
                trial_rows.append(row)

            for ch in range(n_channels):
                row = dict(base)
                row.update(
                    {
                        "channel": ch + 1,
                        "channel_label": channel_labels[ch] if ch < len(channel_labels) else f"Ch{ch+1:02d}",
                        "enPrune_active_fraction": float(active_fraction[ch]),
                        "enPrune_active": bool(active_channels[ch]),
                        "enPrune_flagged": bool(enprune_flagged[ch]),
                        "motion_clean_fraction_tIncChAuto": float(motion_clean_fraction[ch]),
                        "motion_flagged": bool(motion_flagged[ch]) if np.isfinite(motion_clean_fraction[ch]) else True,
                        "raw_intensity_cv_percent": float(raw_cv[ch]) if np.isfinite(raw_cv[ch]) else np.nan,
                        "raw_cv_flagged": bool(raw_cv[ch] > RAW_CV_FLAG_THRESHOLD_PERCENT) if np.isfinite(raw_cv[ch]) else False,
                        "excluded_channel": bool(final_excluded[ch]),
                        "retained_channel": bool(retained[ch]),
                    }
                )
                channel_rows.append(row)

            run_row = dict(base)
            run_row.update(
                {
                    "sampling_frequency_hz": safe_fs(raw_t),
                    "n_channels": n_channels,
                    "n_trials": len(windows),
                    "n_enPrune_flagged_channels": int(np.sum(enprune_flagged)),
                    "n_motion_flagged_channels": int(np.sum(motion_flagged)),
                    "n_excluded_channels_final": int(np.sum(final_excluded)),
                    "percent_retained_channels_final": 100.0 * float(np.mean(retained)),
                    "mean_motion_clean_fraction": float(np.nanmean(motion_clean_fraction)),
                    "median_raw_intensity_cv_percent": float(np.nanmedian(raw_cv)),
                    "n_raw_cv_flagged_channels": int(np.nansum(raw_cv > RAW_CV_FLAG_THRESHOLD_PERCENT)),
                    "mean_trial_clean_fraction": float(np.nanmean(trial_clean_values)),
                    "n_excluded_trials": int(np.sum(trial_excluded_values)),
                    "percent_retained_trials": 100.0 * (1.0 - float(np.mean(trial_excluded_values))) if trial_excluded_values else np.nan,
                }
            )
            run_rows.append(run_row)
            hrf_inputs.append(
                {
                    "base": base,
                    "proc_t": proc_t,
                    "proc_s": proc_s,
                    "dc": dc,
                    "clean_time": clean_time,
                    "retained_channels": retained,
                    "trial_excluded_values": np.asarray(trial_excluded_values, dtype=bool),
                    "windows": windows,
                }
            )
        except Exception as exc:
            time_row["reason"] = f"load_error: {exc}"

        time_rows.append(time_row)
        if i_file % 16 == 0 or i_file == len(raw_files):
            print(f"QC collected: {i_file}/{len(raw_files)}", flush=True)

    time_df = pd.DataFrame(time_rows)
    param_df = pd.DataFrame(param_rows)
    run_df = pd.DataFrame(run_rows)
    channel_df = pd.DataFrame(channel_rows)
    trial_df = pd.DataFrame(trial_rows)

    #time_df.to_csv(out_dir / "timepoint_consistency_check.csv", index=False, encoding="utf-8-sig")
    #time_df.loc[~time_df["timepoints_match"]].to_csv(
    #    out_dir / "timepoint_mismatches.csv", index=False, encoding="utf-8-sig"
    #)
    #param_df.to_csv(out_dir / "preprocessing_parameter_check.csv", index=False, encoding="utf-8-sig")
    #param_df.loc[~param_df["matches_expected"]].to_csv(
    #    out_dir / "preprocessing_parameter_mismatches.csv", index=False, encoding="utf-8-sig"
    #)
    run_df.to_csv(out_dir / "qc_run_summary.csv", index=False, encoding="utf-8-sig")
    channel_df.to_csv(out_dir / "qc_channel_summary.csv", index=False, encoding="utf-8-sig")
    trial_df.to_csv(out_dir / "qc_trial_summary.csv", index=False, encoding="utf-8-sig")

    return time_df, param_df, run_df, channel_df, trial_df, hrf_inputs


def extract_hrf(hrf_inputs, out_dir: Path):
    epoch_time = np.arange(EPOCH_START_S, EPOCH_END_S + 0.5 / TARGET_FS, 1.0 / TARGET_FS)
    baseline_mask = (epoch_time >= BASELINE_START_S) & (epoch_time < BASELINE_END_S)
    hrf_by_run_hbo = {run: [] for run in RUN_CONDITIONS}
    hrf_by_run_hbr = {run: [] for run in RUN_CONDITIONS}
    hrf_info_rows = []

    for item in hrf_inputs:
        base = item["base"]
        run = int(base["run"].split("-")[1])
        t = item["proc_t"]
        dc = item["dc"]
        hbo = dc[:, 0, :] * 1e6
        hbr = dc[:, 1, :] * 1e6
        n_channels = hbo.shape[1]
        retained_channels = item["retained_channels"]
        trial_excluded = item["trial_excluded_values"]
        windows = item["windows"]
        hbo_epochs = []
        hbr_epochs = []
        skipped = 0
        used_trial_flags = []

        for i_trial, (idx0, t0, _mask) in enumerate(windows):
            if i_trial < trial_excluded.size and trial_excluded[i_trial]:
                skipped += 1
                used_trial_flags.append(False)
                continue
            sample_t = t0 + epoch_time
            if sample_t[0] < t[0] or sample_t[-1] > t[-1]:
                skipped += 1
                used_trial_flags.append(False)
                continue
            epoch_hbo = np.full((epoch_time.size, n_channels), np.nan)
            epoch_hbr = np.full((epoch_time.size, n_channels), np.nan)
            for ch in np.where(retained_channels)[0]:
                epoch_hbo[:, ch] = np.interp(sample_t, t, hbo[:, ch], left=np.nan, right=np.nan)
                epoch_hbr[:, ch] = np.interp(sample_t, t, hbr[:, ch], left=np.nan, right=np.nan)
            hbo_base = np.nanmean(epoch_hbo[baseline_mask, :], axis=0)
            hbr_base = np.nanmean(epoch_hbr[baseline_mask, :], axis=0)
            epoch_hbo = epoch_hbo - hbo_base
            epoch_hbr = epoch_hbr - hbr_base
            hbo_epochs.append(epoch_hbo)
            hbr_epochs.append(epoch_hbr)
            used_trial_flags.append(True)

        hrf_info = dict(base)
        hrf_info.update(
            {
                "n_trial_markers": len(windows),
                "n_valid_epochs_used": len(hbo_epochs),
                "n_skipped_epochs": skipped,
                "n_channels_total": n_channels,
                "n_channels_retained": int(np.sum(retained_channels)),
            }
        )
        hrf_info_rows.append(hrf_info)

        if hbo_epochs:
            hrf_by_run_hbo[run].append(np.nanmean(np.stack(hbo_epochs, axis=0), axis=0))
            hrf_by_run_hbr[run].append(np.nanmean(np.stack(hbr_epochs, axis=0), axis=0))

    hrf_info_df = pd.DataFrame(hrf_info_rows)
    hrf_info_df.to_csv(out_dir / "hrf_extraction_summary.csv", index=False, encoding="utf-8-sig")

    rows = []
    for run, (_mode, _difficulty, _pattern, label, _short) in RUN_CONDITIONS.items():
        hbo_arr = np.stack(hrf_by_run_hbo[run], axis=0) if hrf_by_run_hbo[run] else np.empty((0, epoch_time.size, 52))
        hbr_arr = np.stack(hrf_by_run_hbr[run], axis=0) if hrf_by_run_hbr[run] else np.empty((0, epoch_time.size, 52))
        n_channels = hbo_arr.shape[2] if hbo_arr.ndim == 3 else 52
        for ch in range(n_channels):
            for i_t, tt in enumerate(epoch_time):
                for chrom, arr in [("HbO", hbo_arr), ("HbR", hbr_arr)]:
                    vals = arr[:, i_t, ch] if arr.size else np.asarray([])
                    n = int(np.sum(np.isfinite(vals)))
                    mean = float(np.nanmean(vals)) if n else np.nan
                    sem = float(np.nanstd(vals, ddof=1) / math.sqrt(n)) if n > 1 else np.nan
                    rows.append(
                        {
                            "run": f"run-{run:02d}",
                            "condition_label": label,
                            "channel": ch + 1,
                            "time_s": float(tt),
                            "chromophore": chrom,
                            "mean_uM": mean,
                            "sem_uM": sem,
                            "n_dyads": n,
                        }
                    )
    hrf_ts = pd.DataFrame(rows)
    hrf_ts.to_csv(out_dir / "group_hrf_timeseries.csv", index=False, encoding="utf-8-sig")

    metrics = compute_channel_metrics(hrf_by_run_hbo, hrf_by_run_hbr, epoch_time)
    metrics.to_csv(out_dir / "channel_hrf_metrics_by_condition.csv", index=False, encoding="utf-8-sig")
    ranking = rank_representative_channels(metrics)
    ranking.to_csv(out_dir / "channel_representativeness_ranking.csv", index=False, encoding="utf-8-sig")
    make_channelwise_hrf_figures(out_dir, hrf_by_run_hbo, hrf_by_run_hbr, epoch_time)
    make_hbo_peak_overview(out_dir, metrics)
    return hrf_ts, metrics, ranking, hrf_info_df, hrf_by_run_hbo, hrf_by_run_hbr, epoch_time


def compute_channel_metrics(hrf_by_run_hbo, hrf_by_run_hbr, epoch_time: np.ndarray) -> pd.DataFrame:
    task_mask = (epoch_time >= 0) & (epoch_time <= 25)
    late_mask = (epoch_time >= 5) & (epoch_time <= 25)
    rows = []
    for run, (_mode, _difficulty, _pattern, label, _short) in RUN_CONDITIONS.items():
        hbo = np.stack(hrf_by_run_hbo[run], axis=0) if hrf_by_run_hbo[run] else np.empty((0, epoch_time.size, 52))
        hbr = np.stack(hrf_by_run_hbr[run], axis=0) if hrf_by_run_hbr[run] else np.empty((0, epoch_time.size, 52))
        n_channels = hbo.shape[2] if hbo.ndim == 3 else 52
        for ch in range(n_channels):
            hbo_vals = hbo[:, :, ch] if hbo.size else np.empty((0, epoch_time.size))
            hbr_vals = hbr[:, :, ch] if hbr.size else np.empty((0, epoch_time.size))
            n_dyads = int(np.sum(np.isfinite(hbo_vals[:, 0]))) if hbo_vals.size else 0
            hbo_mean = np.nanmean(hbo_vals, axis=0) if hbo_vals.size else np.full(epoch_time.size, np.nan)
            hbr_mean = np.nanmean(hbr_vals, axis=0) if hbr_vals.size else np.full(epoch_time.size, np.nan)
            if np.any(np.isfinite(hbo_mean[task_mask])):
                hbo_peak_idx = int(np.nanargmax(hbo_mean[task_mask]))
                hbo_peak = float(np.nanmax(hbo_mean[task_mask]))
                hbo_peak_time = float(epoch_time[task_mask][hbo_peak_idx])
            else:
                hbo_peak = np.nan
                hbo_peak_time = np.nan
            if np.any(np.isfinite(hbr_mean[task_mask])):
                hbr_min = float(np.nanmin(hbr_mean[task_mask]))
                hbr_min_time = float(epoch_time[task_mask][int(np.nanargmin(hbr_mean[task_mask]))])
            else:
                hbr_min = np.nan
                hbr_min_time = np.nan
            hbo_auc = float(np.trapz(hbo_mean[late_mask], epoch_time[late_mask])) if np.any(np.isfinite(hbo_mean[late_mask])) else np.nan
            hbr_auc = float(np.trapz(hbr_mean[late_mask], epoch_time[late_mask])) if np.any(np.isfinite(hbr_mean[late_mask])) else np.nan
            rows.append(
                {
                    "run": f"run-{run:02d}",
                    "condition_label": label,
                    "channel": ch + 1,
                    "n_dyads": n_dyads,
                    "hbo_peak_uM_0_25s": hbo_peak,
                    "hbo_peak_time_s": hbo_peak_time,
                    "hbr_min_uM_0_25s": hbr_min,
                    "hbr_min_time_s": hbr_min_time,
                    "hbo_auc_uM_s_5_25s": hbo_auc,
                    "hbr_auc_uM_s_5_25s": hbr_auc,
                    "hbo_minus_hbr_peak_separation_uM": hbo_peak - hbr_min
                    if np.isfinite(hbo_peak) and np.isfinite(hbr_min)
                    else np.nan,
                    "canonical_hbo_pos_hbr_neg": bool(hbo_peak > 0 and hbr_min < 0),
                }
            )
    return pd.DataFrame(rows)


def rank_representative_channels(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ch, sub in metrics.groupby("channel"):
        sub = sub.sort_values("run")
        sep = sub["hbo_minus_hbr_peak_separation_uM"].to_numpy(float)
        hbo_peak = sub["hbo_peak_uM_0_25s"].to_numpy(float)
        hbr_min = sub["hbr_min_uM_0_25s"].to_numpy(float)
        peak_time = sub["hbo_peak_time_s"].to_numpy(float)
        n_dyads = sub["n_dyads"].to_numpy(float)
        canonical = np.logical_and(hbo_peak > 0, hbr_min < 0)
        plausible_time = np.logical_and(peak_time >= 5, peak_time <= 25)
        rows.append(
            {
                "channel": int(ch),
                "mean_separation_uM": float(np.nanmean(sep)),
                "min_separation_uM": float(np.nanmin(sep)),
                "mean_hbo_peak_uM": float(np.nanmean(hbo_peak)),
                "mean_hbr_min_uM": float(np.nanmean(hbr_min)),
                "mean_hbo_peak_time_s": float(np.nanmean(peak_time)),
                "canonical_conditions": int(np.sum(canonical)),
                "plausible_peak_time_conditions": int(np.sum(plausible_time)),
                "min_n_dyads_per_condition": int(np.nanmin(n_dyads)),
                "mean_n_dyads_per_condition": float(np.nanmean(n_dyads)),
            }
        )
    ranking = pd.DataFrame(rows)
    ranking["representative_score"] = (
        ranking["mean_separation_uM"].clip(lower=0)
        * (ranking["canonical_conditions"] / 4.0)
        * (ranking["plausible_peak_time_conditions"] / 4.0)
        * (ranking["min_n_dyads_per_condition"] / 32.0)
    )
    ranking = ranking.sort_values(
        ["representative_score", "canonical_conditions", "min_n_dyads_per_condition", "mean_separation_uM"],
        ascending=[False, False, False, False],
    )
    return ranking


def shaded_mean_sem(ax, x: np.ndarray, arr: np.ndarray, color: str, label: str):
    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    ax.plot(x, mean, color=color, lw=1.8, label=label)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)


def make_representative_hrf_figure(out_dir: Path, hrf_ts: pd.DataFrame, channel: int):
    sub_ch = hrf_ts[hrf_ts["channel"].eq(channel)].copy()
    y_min = float(np.nanmin(sub_ch["mean_uM"] - sub_ch["sem_uM"]))
    y_max = float(np.nanmax(sub_ch["mean_uM"] + sub_ch["sem_uM"]))
    y_pad = max(0.15, 0.08 * (y_max - y_min))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    for ax, run in zip(axes.flat, [1, 2, 3, 4]):
        _mode, _difficulty, _pattern, label, _short = RUN_CONDITIONS[run]
        sub = sub_ch[sub_ch["run"].eq(f"run-{run:02d}")]
        hbo = sub[sub["chromophore"].eq("HbO")].sort_values("time_s")
        hbr = sub[sub["chromophore"].eq("HbR")].sort_values("time_s")
        ax.plot(hbo["time_s"], hbo["mean_uM"], color="#C0392B", lw=1.8, label="HbO")
        ax.fill_between(
            hbo["time_s"].to_numpy(),
            (hbo["mean_uM"] - hbo["sem_uM"]).to_numpy(),
            (hbo["mean_uM"] + hbo["sem_uM"]).to_numpy(),
            color="#C0392B",
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(hbr["time_s"], hbr["mean_uM"], color="#2166AC", lw=1.8, label="HbR")
        ax.fill_between(
            hbr["time_s"].to_numpy(),
            (hbr["mean_uM"] - hbr["sem_uM"]).to_numpy(),
            (hbr["mean_uM"] + hbr["sem_uM"]).to_numpy(),
            color="#2166AC",
            alpha=0.18,
            linewidth=0,
        )
        ax.axvspan(0, 20, color="#C7C7C7", alpha=0.18, linewidth=0)
        ax.axvline(0, color="#333333", lw=0.8, ls="--")
        ax.axhline(0, color="#777777", lw=0.7)
        ax.set_title(label)
        ax.set_xlim(EPOCH_START_S, EPOCH_END_S)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(axis="y", color="#000000", alpha=0.10, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Concentration change (uM)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time from trial onset (s)")
    axes.flat[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle(f"Group-averaged task-evoked HRF at channel {channel:02d}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"figure_representative_channel-{channel:02d}_hrf.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"figure_representative_channel-{channel:02d}_hrf.pdf", bbox_inches="tight")
    plt.close(fig)


def make_channelwise_hrf_figures(out_dir: Path, hrf_by_run_hbo, hrf_by_run_hbr, epoch_time: np.ndarray):
    channel_dir = out_dir / "channel_hrf_figures"
    channel_dir.mkdir(exist_ok=True)
    first = next((run for run in RUN_CONDITIONS if hrf_by_run_hbo[run]), None)
    if first is None:
        return
    n_channels = np.stack(hrf_by_run_hbo[first], axis=0).shape[2]
    for ch in range(n_channels):
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.3), sharex=True, sharey=True)
        for ax, run in zip(axes.flat, [1, 2, 3, 4]):
            hbo = np.stack(hrf_by_run_hbo[run], axis=0)[:, :, ch] if hrf_by_run_hbo[run] else np.empty((0, epoch_time.size))
            hbr = np.stack(hrf_by_run_hbr[run], axis=0)[:, :, ch] if hrf_by_run_hbr[run] else np.empty((0, epoch_time.size))
            shaded_mean_sem(ax, epoch_time, hbo, "#C0392B", "HbO")
            shaded_mean_sem(ax, epoch_time, hbr, "#2166AC", "HbR")
            ax.axvspan(0, 20, color="#C7C7C7", alpha=0.18, linewidth=0)
            ax.axvline(0, color="#333333", lw=0.8, ls="--")
            ax.axhline(0, color="#777777", lw=0.7)
            ax.set_title(RUN_CONDITIONS[run][3])
            ax.grid(axis="y", alpha=0.12, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("Concentration change (uM)")
        for ax in axes[-1, :]:
            ax.set_xlabel("Time from trial onset (s)")
        axes.flat[0].legend(frameon=False, loc="upper left", ncol=2)
        fig.suptitle(f"Group-averaged HRF, channel {ch + 1:02d}", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(channel_dir / f"group_average_hrf_channel-{ch + 1:02d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_hbo_peak_overview(out_dir: Path, metrics: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), sharex=True, sharey=True)
    ymax = max(0.1, float(np.nanmax(metrics["hbo_peak_uM_0_25s"]))) * 1.12
    for ax, run in zip(axes.flat, [1, 2, 3, 4]):
        sub = metrics[metrics["run"].eq(f"run-{run:02d}")].sort_values("channel")
        ax.bar(sub["channel"], sub["hbo_peak_uM_0_25s"], color="#C0392B", alpha=0.75)
        ax.set_title(RUN_CONDITIONS[run][3])
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Channel")
        ax.set_ylabel("HbO peak (uM)")
        ax.grid(axis="y", alpha=0.14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Channel-wise HbO peak overview", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "figure_channel_hbo_peak_overview.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "figure_channel_hbo_peak_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def save_table_image(df: pd.DataFrame, out_png: Path, out_pdf: Path, title: str, col_widths: list[float]):
    fig, ax = plt.subplots(figsize=(10.8, 0.55 * (len(df) + 1) + 0.9))
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
    table.scale(1.0, 1.42)
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


def make_qc_tables_and_figures(out_dir: Path, time_df, param_df, run_df, channel_df, trial_df, ranking):
    def pct(n, denom):
        return 100.0 * float(n) / float(denom) if denom else np.nan

    channel_run_retained = 100 * channel_df.groupby(["dyad", "run"])["retained_channel"].mean()
    trial_run_retained = 100 * (1 - trial_df.groupby(["dyad", "run"])["excluded_trial"].mean().astype(float))
    n_files = len(time_df)
    n_time_ok = int(time_df["timepoints_match"].sum())
    n_param_ok = int(param_df.groupby(["dyad", "run"])["matches_expected"].all().sum()) if len(param_df) else 0
    final_excluded = int(channel_df["excluded_channel"].sum())
    enprune_excluded = int(channel_df["enPrune_flagged"].sum())
    motion_excluded = int(channel_df["motion_flagged"].sum())
    trial_excluded = int(trial_df["excluded_trial"].sum())

    concise = pd.DataFrame(
        [
            {
                "QC domain": "File consistency",
                "Criterion": "Identical raw/processed time points",
                "Result": f"{n_time_ok}/{n_files} matched",
            },
            {
                "QC domain": "Preprocessing check",
                "Criterion": "Expected enPrune, motion, spline, band-pass, OD2Conc, block average parameters",
                "Result": f"{n_param_ok}/{n_files} files matched all parameters",
            },
            {
                "QC domain": "Initial channel pruning",
                "Criterion": "enPruneChannels MeasListAct retained",
                "Result": f"{len(channel_df) - enprune_excluded}/{len(channel_df)} retained ({100-pct(enprune_excluded, len(channel_df)):.2f}%)",
            },
            {
                "QC domain": "Motion quality",
                "Criterion": "tIncChAuto channel clean fraction >= 0.80",
                "Result": f"{len(channel_df) - motion_excluded}/{len(channel_df)} retained ({100-pct(motion_excluded, len(channel_df)):.2f}%)",
            },
            {
                "QC domain": "Final channel retention",
                "Criterion": "enPrune retained and motion clean fraction >= 0.80",
                "Result": f"{channel_run_retained.median():.1f}% retained/run; {final_excluded}/{len(channel_df)} flagged ({pct(final_excluded, len(channel_df)):.2f}%)",
            },
            {
                "QC domain": "Trial quality",
                "Criterion": "Mean tIncChAuto across enPrune-retained channels >= 0.80",
                "Result": f"{trial_run_retained.median():.1f}% retained/run; {trial_excluded}/{len(trial_df)} flagged ({pct(trial_excluded, len(trial_df)):.2f}%)",
            },
            {
                "QC domain": "Raw signal variability",
                "Criterion": "Raw optical-intensity CV",
                "Result": f"Median CV = {channel_df['raw_intensity_cv_percent'].median():.2f}%",
            },
        ]
    )
    concise.to_csv(out_dir / "table_qc_concise_summary.csv", index=False, encoding="utf-8-sig")
    save_table_image(
        concise,
        out_dir / "table_qc_concise_summary.png",
        out_dir / "table_qc_concise_summary.pdf",
        "fNIRS signal-quality summary",
        [0.20, 0.45, 0.35],
    )

    condition_rows = []
    for run, (_mode, _difficulty, _pattern, _label, short) in RUN_CONDITIONS.items():
        ch = channel_df[channel_df["run"].eq(f"run-{run:02d}")]
        tr = trial_df[trial_df["run"].eq(f"run-{run:02d}")]
        ch_rate = 100 * ch.groupby("dyad")["retained_channel"].mean()
        tr_rate = 100 * (1 - tr.groupby("dyad")["excluded_trial"].mean())
        condition_rows.append(
            {
                "Condition": short,
                "Run": f"run-{run:02d}",
                "N dyads": ch["dyad"].nunique(),
                "enPrune flagged": f"{int(ch['enPrune_flagged'].sum())}/{len(ch)} ({pct(ch['enPrune_flagged'].sum(), len(ch)):.1f}%)",
                "Motion flagged": f"{int(ch['motion_flagged'].sum())}/{len(ch)} ({pct(ch['motion_flagged'].sum(), len(ch)):.1f}%)",
                "Final ch retained": f"{ch_rate.median():.1f}%",
                "Trial retained": f"{tr_rate.median():.1f}%",
                "Trial flagged": f"{int(tr['excluded_trial'].sum())}/{len(tr)} ({pct(tr['excluded_trial'].sum(), len(tr)):.1f}%)",
            }
        )
    condition_qc = pd.DataFrame(condition_rows)
    condition_qc.to_csv(out_dir / "table_qc_by_condition.csv", index=False, encoding="utf-8-sig")
    save_table_image(
        condition_qc,
        out_dir / "table_qc_by_condition.png",
        out_dir / "table_qc_by_condition.pdf",
        "Condition-wise fNIRS quality-control summary",
        [0.13, 0.09, 0.08, 0.17, 0.17, 0.13, 0.12, 0.11],
    )

    top = ranking.head(10).copy()
    top_display = top[
        [
            "channel",
            "representative_score",
            "canonical_conditions",
            "plausible_peak_time_conditions",
            "min_n_dyads_per_condition",
            "mean_separation_uM",
            "mean_hbo_peak_uM",
            "mean_hbr_min_uM",
        ]
    ].copy()
    for col in ["representative_score", "mean_separation_uM", "mean_hbo_peak_uM", "mean_hbr_min_uM"]:
        top_display[col] = top_display[col].map(lambda x: f"{x:.3f}")
    top_display.to_csv(out_dir / "table_representative_channel_candidates.csv", index=False, encoding="utf-8-sig")
    save_table_image(
        top_display,
        out_dir / "table_representative_channel_candidates.png",
        out_dir / "table_representative_channel_candidates.pdf",
        "Candidate representative channels for HRF visualization",
        [0.08, 0.16, 0.14, 0.18, 0.16, 0.14, 0.12, 0.12],
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    ax = axes[0, 0]
    ax.hist(channel_df["enPrune_active_fraction"], bins=np.linspace(0, 1, 11), color="#4C78A8", alpha=0.85)
    ax.set_title("enPruneChannels active fraction")
    ax.set_xlabel("Active wavelength fraction")
    ax.set_ylabel("Channel-run count")
    ax = axes[0, 1]
    ax.hist(channel_df["motion_clean_fraction_tIncChAuto"], bins=np.linspace(0, 1, 31), color="#59A14F", alpha=0.85)
    ax.axvline(CHANNEL_MOTION_CLEAN_THRESHOLD, color="#D62728", ls="--", lw=1.3)
    ax.set_title("Motion clean fraction")
    ax.set_xlabel("Clean fraction")
    ax.set_ylabel("Channel-run count")
    ax = axes[1, 0]
    vals = trial_df["clean_fraction_active_channels"].to_numpy(float)
    ax.hist(vals[np.isfinite(vals)], bins=np.linspace(0, 1, 31), color="#F28E2B", alpha=0.85)
    ax.axvline(TRIAL_CLEAN_THRESHOLD, color="#D62728", ls="--", lw=1.3)
    ax.set_title("Trial clean fraction")
    ax.set_xlabel("Clean fraction")
    ax.set_ylabel("Trial count")
    ax = axes[1, 1]
    by_ch = channel_df.groupby("channel").agg(
        retained=("retained_channel", "mean"),
        enprune=("enPrune_flagged", "mean"),
        motion=("motion_flagged", "mean"),
    )
    ax.plot(by_ch.index, 100 * by_ch["retained"], color="#333333", lw=1.5, label="Retained")
    ax.plot(by_ch.index, 100 * by_ch["enprune"], color="#4C78A8", lw=1.0, label="enPrune flagged")
    ax.plot(by_ch.index, 100 * by_ch["motion"], color="#59A14F", lw=1.0, label="Motion flagged")
    ax.set_title("QC rate by channel")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Run observations (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="best")
    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.14)
    fig.suptitle("fNIRS signal quality and exclusion after enPruneChannels", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "figure_qc_overview.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "figure_qc_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    return concise, condition_qc


def make_summary(out_dir: Path, time_df, param_df, run_df, channel_df, trial_df, hrf_info_df, metrics, ranking):
    n_files = len(time_df)
    n_time_ok = int(time_df["timepoints_match"].sum())
    n_param_ok = int(param_df.groupby(["dyad", "run"])["matches_expected"].all().sum()) if len(param_df) else 0
    n_channel = len(channel_df)
    n_trial = len(trial_df)
    enprune_flagged = int(channel_df["enPrune_flagged"].sum())
    motion_flagged = int(channel_df["motion_flagged"].sum())
    final_flagged = int(channel_df["excluded_channel"].sum())
    trial_flagged = int(trial_df["excluded_trial"].sum())
    channel_retained_run = 100 * channel_df.groupby(["dyad", "run"])["retained_channel"].mean()
    trial_retained_run = 100 * (1 - trial_df.groupby(["dyad", "run"])["excluded_trial"].mean())
    selected_channel = int(ranking.iloc[0]["channel"])
    selected_metrics = metrics[metrics["channel"].eq(selected_channel)].sort_values("run")

    peak_parts = []
    for _, row in selected_metrics.iterrows():
        peak_parts.append(
            f"{row['condition_label']}: HbO peak {row['hbo_peak_uM_0_25s']:.2f} uM at "
            f"{row['hbo_peak_time_s']:.1f} s; HbR minimum {row['hbr_min_uM_0_25s']:.2f} uM"
        )

    summary = {
        "n_files": int(n_files),
        "n_timepoint_matched_files": n_time_ok,
        "n_files_all_expected_params": n_param_ok,
        "n_channel_observations": int(n_channel),
        "n_trial_observations": int(n_trial),
        "n_enPrune_flagged_channel_observations": enprune_flagged,
        "percent_enPrune_flagged_channel_observations": 100.0 * enprune_flagged / n_channel if n_channel else np.nan,
        "n_motion_flagged_channel_observations": motion_flagged,
        "percent_motion_flagged_channel_observations": 100.0 * motion_flagged / n_channel if n_channel else np.nan,
        "n_final_excluded_channel_observations": final_flagged,
        "percent_final_excluded_channel_observations": 100.0 * final_flagged / n_channel if n_channel else np.nan,
        "median_retained_channel_rate_per_run_percent": float(channel_retained_run.median()),
        "n_excluded_trial_observations": trial_flagged,
        "percent_excluded_trial_observations": 100.0 * trial_flagged / n_trial if n_trial else np.nan,
        "median_retained_trial_rate_per_run_percent": float(trial_retained_run.median()),
        "median_raw_intensity_cv_percent": float(channel_df["raw_intensity_cv_percent"].median()),
        "selected_representative_channel": selected_channel,
        "selected_channel_metrics": selected_metrics.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

def main():
    out_dir = REPORT_ROOT / f"technical_validation_enprune_qc_hrf_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    time_df, param_df, run_df, channel_df, trial_df, hrf_inputs = collect_qc(out_dir)
    hrf_ts, metrics, ranking, hrf_info_df, hrf_by_run_hbo, hrf_by_run_hbr, epoch_time = extract_hrf(hrf_inputs, out_dir)
    selected_channel = int(ranking.iloc[0]["channel"])
    make_representative_hrf_figure(out_dir, hrf_ts, selected_channel)
    make_qc_tables_and_figures(out_dir, time_df, param_df, run_df, channel_df, trial_df, ranking)
    summary = make_summary(out_dir, time_df, param_df, run_df, channel_df, trial_df, hrf_info_df, metrics, ranking)
    print("Analysis complete.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
