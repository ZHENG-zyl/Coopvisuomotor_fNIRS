from __future__ import annotations

import csv
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
from scipy import stats

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

SCRIPT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("COOPVM_DATASET_ROOT", str(SCRIPT_ROOT.parent)))
REPORT_ROOT = Path(
    os.environ.get(
        "COOPVM_REPORT_ROOT",
        str(DATASET_ROOT / "derivatives" / "technical-validation" / "reproduced"),
    )
)

CONDITIONS = {
    1: {
        "mode": "leader-follower",
        "difficulty": "easy",
        "pattern": "carrot",
        "condition_label": "Leader-follower / carrot",
    },
    2: {
        "mode": "leader-follower",
        "difficulty": "hard",
        "pattern": "wave",
        "condition_label": "Leader-follower / wave",
    },
    3: {
        "mode": "egalitarian",
        "difficulty": "easy",
        "pattern": "carrot",
        "condition_label": "Egalitarian / carrot",
    },
    4: {
        "mode": "egalitarian",
        "difficulty": "hard",
        "pattern": "wave",
        "condition_label": "Egalitarian / wave",
    },
}

METRICS = ["PositionScore", "ForceScore", "TotalScore", "CompleteTime_s"]
METRIC_LABELS = {
    "PositionScore": "Position score",
    "ForceScore": "Force score",
    "TotalScore": "Total score",
    "CompleteTime_s": "Completion time",
}
METRIC_YLABELS = {
    "PositionScore": "Score",
    "ForceScore": "Score",
    "TotalScore": "Score",
    "CompleteTime_s": "Time (s)",
}


def fdr_bh(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    q[valid] = out
    return q


def read_trial_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    files = sorted(DATASET_ROOT.glob("dyad-*/beh/*_beh-trials.csv"))
    if not files:
        raise FileNotFoundError(f"No behavioral trial files found under {DATASET_ROOT}")

    for path in files:
        match = re.match(r"dyad-(\d+)_task-coopvm_run-(\d+)_beh-trials\.csv", path.name)
        if not match:
            raise ValueError(f"Unexpected filename: {path.name}")

        dyad = int(match.group(1))
        run = int(match.group(2))
        condition = CONDITIONS[run]

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out = {
                    "Dyad": dyad,
                    "Run": run,
                    "Mode": condition["mode"],
                    "Difficulty": condition["difficulty"],
                    "Pattern": condition["pattern"],
                    "ConditionLabel": condition["condition_label"],
                    "Number": int(row["Number"]),
                    "ForceScore": float(row["ForceScore"]),
                    "PositionScore": float(row["PositionScore"]),
                    "TotalScore": float(row["TotalScore"]),
                    "CompleteTime_s": float(row["CompleteTime_s"]),
                    "SourceFile": str(path),
                }
                rows.append(out)

    return pd.DataFrame(rows)


def paired_effect(values: np.ndarray) -> dict[str, float]:
    values = values[~np.isnan(values)]
    n = len(values)
    df = n - 1
    mean_diff = values.mean()
    sd_diff = values.std(ddof=1)
    sem = sd_diff / math.sqrt(n)
    t_value = mean_diff / sem
    p_value = stats.t.sf(abs(t_value), df) * 2
    tcrit = stats.t.ppf(0.975, df)
    f_value = t_value**2
    return {
        "N": n,
        "MeanDifference": mean_diff,
        "SDDifference": sd_diff,
        "CI95_Low": mean_diff - tcrit * sem,
        "CI95_High": mean_diff + tcrit * sem,
        "t": t_value,
        "df": df,
        "F_1_df": f_value,
        "p_uncorrected": p_value,
        "Cohen_dz": mean_diff / sd_diff,
        "PartialEtaSquared": f_value / (f_value + df),
    }


def main() -> None:
    output_root = REPORT_ROOT / f"behavioral_validation_{datetime.now():%Y%m%d_%H%M%S}"
    output_root.mkdir(parents=True, exist_ok=True)

    trial_data = read_trial_data()

    dyad_condition_means = (
        trial_data.groupby(
            ["Dyad", "Run", "Mode", "Difficulty", "Pattern", "ConditionLabel"],
            as_index=False,
        )[METRICS]
        .mean()
        .sort_values(["Dyad", "Run"])
    )
    trial_n = (
        trial_data.groupby(["Dyad", "Run"], as_index=False)
        .agg(NTrials=("Number", "count"))
        .sort_values(["Dyad", "Run"])
    )
    dyad_condition_means = dyad_condition_means.merge(trial_n, on=["Dyad", "Run"])
    dyad_condition_means.to_csv(output_root / "dyad_condition_means.csv", index=False)

    desc_base = (
        dyad_condition_means[
            ["Run", "Mode", "Difficulty", "Pattern", "ConditionLabel"]
        ]
        .drop_duplicates()
        .sort_values("Run")
        .reset_index(drop=True)
    )
    descriptive = desc_base.copy()
    grouped = dyad_condition_means.groupby(
        ["Run", "Mode", "Difficulty", "Pattern", "ConditionLabel"], sort=False
    )
    for metric in METRICS:
        agg = grouped[metric].agg(["count", "mean", "std"]).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])
        agg["ci95_halfwidth"] = [
            stats.t.ppf(0.975, n - 1) * sem
            for n, sem in zip(agg["count"], agg["sem"])
        ]
        descriptive = descriptive.merge(
            agg[
                [
                    "Run",
                    f"{metric}_N",
                    f"{metric}_Mean",
                    f"{metric}_SD",
                    f"{metric}_SEM",
                    f"{metric}_CI95_Low",
                    f"{metric}_CI95_High",
                ]
            ]
            if False
            else pd.DataFrame(
                {
                    "Run": agg["Run"],
                    f"{metric}_N": agg["count"],
                    f"{metric}_Mean": agg["mean"],
                    f"{metric}_SD": agg["std"],
                    f"{metric}_SEM": agg["sem"],
                    f"{metric}_CI95_Low": agg["mean"] - agg["ci95_halfwidth"],
                    f"{metric}_CI95_High": agg["mean"] + agg["ci95_halfwidth"],
                }
            ),
            on="Run",
        )
    descriptive.to_csv(output_root / "condition_descriptive_statistics.csv", index=False)

    stat_rows = []
    for metric in METRICS:
        wide = dyad_condition_means.pivot(index="Dyad", columns="Run", values=metric)
        y = wide[[1, 2, 3, 4]].to_numpy(dtype=float)
        effects = {
            "Difficulty": (
                "hard/wave - easy/carrot",
                np.nanmean(y[:, [1, 3]], axis=1) - np.nanmean(y[:, [0, 2]], axis=1),
            ),
            "Mode": (
                "egalitarian - leader-follower",
                np.nanmean(y[:, [2, 3]], axis=1) - np.nanmean(y[:, [0, 1]], axis=1),
            ),
            "Difficulty x Mode": (
                "(egalitarian hard - egalitarian easy) - (leader-follower hard - leader-follower easy)",
                (y[:, 3] - y[:, 2]) - (y[:, 1] - y[:, 0]),
            ),
        }
        for effect, (contrast, values) in effects.items():
            row = {"Metric": metric, "Effect": effect, "Contrast": contrast}
            row.update(paired_effect(values))
            stat_rows.append(row)

    stats_df = pd.DataFrame(stat_rows)
    stats_df["p_fdr"] = fdr_bh(stats_df["p_uncorrected"].to_list())
    stats_df = stats_df[
        [
            "Metric",
            "Effect",
            "Contrast",
            "N",
            "MeanDifference",
            "SDDifference",
            "CI95_Low",
            "CI95_High",
            "t",
            "df",
            "F_1_df",
            "p_uncorrected",
            "p_fdr",
            "Cohen_dz",
            "PartialEtaSquared",
        ]
    ]
    stats_df.to_csv(output_root / "within_dyad_effect_statistics.csv", index=False)

    make_figure(dyad_condition_means, descriptive, output_root)

    print(f"Output folder: {output_root}")
    print(f"Trial rows: {len(trial_data)}")
    print(f"Dyad-condition rows: {len(dyad_condition_means)}")


def make_figure(dyad_condition_means: pd.DataFrame, descriptive: pd.DataFrame, output_root: Path) -> None:
    mode_order = ["leader-follower", "egalitarian"]
    difficulty_order = ["easy", "hard"]
    mode_labels = ["Leader-follower", "Egalitarian"]
    difficulty_labels = ["Carrot/easy", "Wave/hard"]
    colors = ["#ED7C2E", "#207DB8"]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.7))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88, wspace=0.28, hspace=0.38)
    axes = axes.ravel()

    rng = np.random.default_rng(20260703)
    width = 0.34
    x = np.arange(len(mode_order))

    for ax, metric in zip(axes, METRICS):
        mean_matrix = np.zeros((2, 2))
        sd_matrix = np.zeros((2, 2))

        for i_mode, mode in enumerate(mode_order):
            for i_diff, diff in enumerate(difficulty_order):
                row = descriptive[
                    (descriptive["Mode"] == mode) & (descriptive["Difficulty"] == diff)
                ].iloc[0]
                mean_matrix[i_mode, i_diff] = row[f"{metric}_Mean"]
                sd_matrix[i_mode, i_diff] = row[f"{metric}_SD"]

        for i_diff, diff in enumerate(difficulty_order):
            offset = (i_diff - 0.5) * width
            ax.bar(
                x + offset,
                mean_matrix[:, i_diff],
                width=width,
                yerr=sd_matrix[:, i_diff],
                color=colors[i_diff],
                edgecolor="#262626",
                linewidth=0.7,
                capsize=3,
                label=difficulty_labels[i_diff],
            )
            for i_mode, mode in enumerate(mode_order):
                values = dyad_condition_means.loc[
                    (dyad_condition_means["Mode"] == mode)
                    & (dyad_condition_means["Difficulty"] == diff),
                    metric,
                ].to_numpy()
                jitter = rng.uniform(-0.035, 0.035, len(values))
                ax.scatter(
                    np.full_like(values, x[i_mode] + offset) + jitter,
                    values,
                    s=9,
                    color="#202020",
                    alpha=0.25,
                    linewidths=0,
                    zorder=3,
                )

        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.set_ylabel(METRIC_YLABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels)
        if metric == "CompleteTime_s":
            ax.set_ylim(0, 18)
        else:
            ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.savefig(output_root / "behavioral_validation_figure.png", dpi=600)
    fig.savefig(output_root / "behavioral_validation_figure.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
