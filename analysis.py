"""
analysis.py  —  Stroop A/B Test: Between-Subject Statistical Analysis
======================================================================
Loads all participant CSV files collected from experiment.py,
aggregates to participant-level summaries, and runs between-subject
statistical tests comparing Group A (congruent) vs Group B (incongruent).

USAGE:
    python analysis.py --data ./data/

    Place all participant CSVs (stroop_A_*.csv, stroop_B_*.csv) in a folder
    and pass the folder path via --data. Defaults to ./data/ if not specified.

OUTPUT:
    - Printed statistical summary in the terminal
    - stroop_results.png  (charts saved to working directory)
    - stroop_participant_summary.csv  (per-participant aggregated data)

REQUIREMENTS:
    pip install pandas numpy scipy matplotlib
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


# ── Command-line arguments ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Between-subject statistical analysis for the Stroop A/B experiment."
)
parser.add_argument(
    "--data",
    type=str,
    default="./data/",
    help="Path to folder containing participant CSV files (default: ./data/)",
)
parser.add_argument(
    "--out",
    type=str,
    default=".",
    help="Output directory for charts and summary CSV (default: current directory)",
)
args = parser.parse_args()


# ── Load participant CSV files ─────────────────────────────────────────────────
csv_files = glob.glob(os.path.join(args.data, "*.csv"))

if not csv_files:
    print(f"ERROR: No CSV files found in '{args.data}'.")
    print("Make sure participant CSVs are in the data folder and try again.")
    sys.exit(1)

dfs = []
skipped = []
for fpath in csv_files:
    try:
        df_p = pd.read_csv(fpath)
        required_cols = {"user_id", "group", "correct", "reaction_time_ms"}
        if required_cols.issubset(df_p.columns):
            dfs.append(df_p)
        else:
            skipped.append(os.path.basename(fpath))
    except Exception as e:
        skipped.append(os.path.basename(fpath))

if skipped:
    print(f"WARNING: Skipped {len(skipped)} file(s) with missing columns: {skipped}")

if not dfs:
    print("ERROR: No valid CSV files could be loaded. Check file format.")
    sys.exit(1)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nLoaded {len(dfs)} participant file(s)  |  {len(df_all)} total trials")


# ── Participant-level aggregation ──────────────────────────────────────────────
# Each participant contributes ONE data point: their mean RT and accuracy.
# This is the correct unit of analysis for a between-subject design.
# Using trial-level data directly would cause pseudoreplication.

def aggregate_participant(grp: pd.DataFrame) -> pd.Series:
    """Summarise one participant's data into a single row."""
    correct_trials = grp[grp["correct"] == True]
    return pd.Series({
        "group":        grp["group"].iloc[0],
        "n_trials":     len(grp),
        "mean_rt_ms":   correct_trials["reaction_time_ms"].mean(),
        "median_rt_ms": correct_trials["reaction_time_ms"].median(),
        "sd_rt_ms":     correct_trials["reaction_time_ms"].std(ddof=1),
        "accuracy_pct": grp["correct"].mean() * 100,
        "n_correct":    int(grp["correct"].sum()),
    })

df_participants = (
    df_all
    .groupby("user_id")
    .apply(aggregate_participant)
    .reset_index()
)

# ── Outlier removal: ±3 SD within each group ──────────────────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    mu, sd = df["mean_rt_ms"].mean(), df["mean_rt_ms"].std()
    return df[(df["mean_rt_ms"] > mu - 3 * sd) & (df["mean_rt_ms"] < mu + 3 * sd)]

df_clean   = df_participants.groupby("group", group_keys=False).apply(remove_outliers)
n_removed  = len(df_participants) - len(df_clean)

grp_a = df_clean[df_clean["group"] == "A"]
grp_b = df_clean[df_clean["group"] == "B"]

a_rt  = grp_a["mean_rt_ms"].values
b_rt  = grp_b["mean_rt_ms"].values
a_acc = grp_a["accuracy_pct"].values
b_acc = grp_b["accuracy_pct"].values


# ── Statistical tests ──────────────────────────────────────────────────────────

# 1. Normality — Shapiro-Wilk (only valid for n < 50 per group)
_, p_norm_a = scipy_stats.shapiro(a_rt)
_, p_norm_b = scipy_stats.shapiro(b_rt)
is_normal   = p_norm_a > 0.05 and p_norm_b > 0.05

# 2. Reaction Time — Welch's t-test (does not assume equal variances)
#    Unit of analysis: per-participant mean RT (correct trials only)
t_stat, p_rt = scipy_stats.ttest_ind(b_rt, a_rt, equal_var=False)

# 3. Effect size — Cohen's d
pooled_sd = np.sqrt(
    (np.std(a_rt, ddof=1) ** 2 + np.std(b_rt, ddof=1) ** 2) / 2
)
cohens_d = (np.mean(b_rt) - np.mean(a_rt)) / pooled_sd

# 4. Accuracy — Mann-Whitney U (one-tailed: B accuracy < A accuracy)
u_stat, p_acc = scipy_stats.mannwhitneyu(b_acc, a_acc, alternative="less")


# ── Print results summary ──────────────────────────────────────────────────────
SEP  = "=" * 62
SEP2 = "-" * 62

print(f"\n{SEP}")
print("  STROOP A/B TEST — Between-Subject Statistical Analysis")
print(SEP)
print(f"  Participants loaded   : {len(df_participants)}")
print(f"  Outliers removed (±3σ): {n_removed}")
print(f"  Final sample          : {len(df_clean)}  "
      f"(Group A: {len(grp_a)}, Group B: {len(grp_b)})")
print(f"\n  {'Group':<22} {'n':>4} {'Mean RT':>10} {'SD RT':>9} {'Accuracy':>10}")
print("  " + SEP2)

for label, rts, accs, n in [
    ("A — Congruent (ctrl)",   a_rt, a_acc, len(grp_a)),
    ("B — Incongruent (trt)",  b_rt, b_acc, len(grp_b)),
]:
    print(
        f"  {label:<22} {n:>4} {np.mean(rts):>9.1f}ms"
        f" {np.std(rts, ddof=1):>8.1f}ms {np.mean(accs):>9.1f}%"
    )

diff      = np.mean(b_rt) - np.mean(a_rt)
direction = "slower" if diff > 0 else "faster"
print(f"\n  RT Difference (B − A) : {diff:+.1f} ms ({direction} incongruent)")

sig   = lambda p: "*** SIGNIFICANT (p < 0.05)" if p < 0.05 else "not significant (p >= 0.05)"
d_lbl = "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"

print(f"\n  Normality (Shapiro-Wilk per group)")
print(f"    Group A: p = {p_norm_a:.3f}  |  Group B: p = {p_norm_b:.3f}"
      f"  ->  {'normal' if is_normal else 'non-normal — Mann-Whitney U may be preferred'}")

print(f"\n  Reaction Time  ·  Welch's t-test on participant-level mean RTs")
print(f"    t = {t_stat:.3f},  p = {p_rt:.4f}  —  {sig(p_rt)}")
print(f"    Cohen's d = {cohens_d:.3f}  ({d_lbl} effect)")

print(f"\n  Accuracy  ·  Mann-Whitney U (one-tailed: B < A)")
print(f"    U = {u_stat:.1f},  p = {p_acc:.4f}  —  {sig(p_acc)}")

decision = (
    "REJECT H0  —  Stroop effect confirmed"
    if p_rt < 0.05
    else "Fail to reject H0  —  no significant RT difference (alpha = 0.05)"
)
print(f"\n  Decision: {decision}")
print(SEP)


# ── Save participant summary CSV ───────────────────────────────────────────────
summary_path = os.path.join(args.out, "stroop_participant_summary.csv")
df_clean.to_csv(summary_path, index=False)
print(f"\n  Participant summary saved -> {summary_path}")


# ── Visualisations ─────────────────────────────────────────────────────────────
PAL   = {"A": "#43A047", "B": "#E53935"}
XLBLS = ["Group A\n(Congruent)", "Group B\n(Incongruent)"]

fig  = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor("white")
fig.suptitle(
    f"Stroop A/B Test  ·  Between-Subject  ·  "
    f"n = {len(df_clean)} (A={len(grp_a)}, B={len(grp_b)})",
    fontsize=14, fontweight="bold", y=1.01,
)
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# ── Plot 1: Mean RT bar + SEM ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
means = [np.mean(a_rt), np.mean(b_rt)]
sems  = [scipy_stats.sem(a_rt), scipy_stats.sem(b_rt)]
bars  = ax.bar(XLBLS, means, color=[PAL["A"], PAL["B"]],
               width=0.5, edgecolor="white")
ax.errorbar(XLBLS, means, yerr=sems, fmt="none",
            color="#333", capsize=6, linewidth=2)
if p_rt < 0.05:
    y_sig = max(means) + max(sems) + 25
    ax.plot([0, 1], [y_sig, y_sig], color="#333", linewidth=1.2)
    ax.text(0.5, y_sig + 4, "***", ha="center", fontsize=14)
ax.set_title("Mean RT (±SEM)", fontweight="bold", fontsize=11, pad=10)
ax.set_ylabel("RT (ms)")
ax.set_facecolor("#FAFAFA")
ax.spines[["top", "right"]].set_visible(False)
for bar, v in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

# ── Plot 2: Violin + individual participant dots ───────────────────────────────
ax = fig.add_subplot(gs[0, 1])
vp = ax.violinplot([a_rt, b_rt], positions=[1, 2],
                   showmedians=True, showextrema=False)
for body, c in zip(vp["bodies"], [PAL["A"], PAL["B"]]):
    body.set_facecolor(c)
    body.set_alpha(0.55)
vp["cmedians"].set_color("white")
vp["cmedians"].set_linewidth(2.5)
for i, (pts, c) in enumerate(zip([a_rt, b_rt], [PAL["A"], PAL["B"]]), start=1):
    jitter = np.random.normal(i, 0.06, size=len(pts))
    ax.scatter(jitter, pts, color=c, alpha=0.7, s=40, zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels(XLBLS)
ax.set_title("RT per Participant", fontweight="bold", fontsize=11, pad=10)
ax.set_ylabel("Mean RT (ms)")
ax.set_facecolor("#FAFAFA")
ax.spines[["top", "right"]].set_visible(False)

# ── Plot 3: Accuracy bar ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
acc_means = [np.mean(a_acc), np.mean(b_acc)]
acc_sems  = [scipy_stats.sem(a_acc), scipy_stats.sem(b_acc)]
bars = ax.bar(XLBLS, acc_means, color=[PAL["A"], PAL["B"]],
              width=0.5, edgecolor="white")
ax.errorbar(XLBLS, acc_means, yerr=acc_sems, fmt="none",
            color="#333", capsize=6, linewidth=2)
ax.set_title("Accuracy (±SEM)", fontweight="bold", fontsize=11, pad=10)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 115)
ax.set_facecolor("#FAFAFA")
ax.spines[["top", "right"]].set_visible(False)
for bar, v in zip(bars, acc_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

# ── Plot 4: Overlapping RT histograms ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0:2])
all_rts = np.concatenate([a_rt, b_rt])
bins = np.linspace(all_rts.min() - 50, all_rts.max() + 50, 25)
ax.hist(a_rt, bins=bins, color=PAL["A"], alpha=0.6,
        label=f"Group A — Congruent (n={len(grp_a)})", edgecolor="white")
ax.hist(b_rt, bins=bins, color=PAL["B"], alpha=0.6,
        label=f"Group B — Incongruent (n={len(grp_b)})", edgecolor="white")
ax.axvline(np.mean(a_rt), color=PAL["A"], linewidth=2, linestyle="--", alpha=0.9)
ax.axvline(np.mean(b_rt), color=PAL["B"], linewidth=2, linestyle="--", alpha=0.9)
ax.set_title("Distribution of Participant Mean RTs", fontweight="bold", fontsize=11, pad=10)
ax.set_xlabel("Mean RT (ms)")
ax.set_ylabel("Number of participants")
ax.legend(fontsize=10)
ax.set_facecolor("#FAFAFA")
ax.spines[["top", "right"]].set_visible(False)

# ── Plot 5: Statistics summary box ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
summary_text = (
    f"Statistical Summary\n"
    f"{'─' * 28}\n"
    f"Welch t-test (RT)\n"
    f"  t = {t_stat:.3f}\n"
    f"  p = {p_rt:.4f}  {'*' if p_rt < 0.05 else 'ns'}\n"
    f"  Cohen's d = {cohens_d:.3f} ({d_lbl})\n\n"
    f"Mann-Whitney U (Accuracy)\n"
    f"  U = {u_stat:.0f}\n"
    f"  p = {p_acc:.4f}  {'*' if p_acc < 0.05 else 'ns'}\n\n"
    f"RT Difference (B - A)\n"
    f"  {diff:+.1f} ms\n\n"
    f"{'─' * 28}\n"
    f"{'REJECT H0' if p_rt < 0.05 else 'Fail to reject H0'}"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))

plt.tight_layout()
chart_path = os.path.join(args.out, "stroop_results.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Chart saved -> {chart_path}\n")
