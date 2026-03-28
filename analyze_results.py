#!/usr/bin/env python3
"""
analyze_results.py
------------------
Analyze and plot results from run_experiments.py.

Produces:
  - phase1_summary.csv       MVS mean +/- std, 95% CI per (env, algo, alpha)
  - phase1_monotonicity.csv   Spearman rho of MVS vs alpha per (env, algo)
  - phase2_summary.csv        Reward mean +/- std per (env, algo, alpha)
  - phase2_degradation.csv    Reward ratio vs clean baseline
  - Plots: MVS vs alpha, reward vs alpha, MVS vs reward scatter

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir results/final
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(filepath):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_phase1(results_dir):
    """Load phase1 MVS results into a DataFrame."""
    path = results_dir / "phase1_mvs.jsonl"
    records = load_jsonl(path)
    df = pd.DataFrame([r for r in records if r.get("status") == "ok"])
    return df


def load_phase2(results_dir):
    """Load phase2 reward results into a DataFrame."""
    path = results_dir / "phase2_reward.jsonl"
    records = load_jsonl(path)
    df = pd.DataFrame([r for r in records if r.get("status") == "ok"])
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ci_95(values):
    """Compute 95% CI half-width using t-distribution."""
    n = len(values)
    if n < 2:
        return 0.0
    se = np.std(values, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    return t_crit * se


def benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.empty(n)
    # Step up from largest to smallest
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_p[i] * n / rank,
        )
    return np.clip(adjusted, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Phase 1 analysis
# ---------------------------------------------------------------------------

def analyze_phase1(df, results_dir):
    """Produce phase1 summary CSV and monotonicity CSV."""
    # Filter to trained-policy results only (not random)
    df_trained = df[df["phase"] == "phase1b"].copy()

    # Summary: mean, std, CI per (env, algo, alpha)
    rows = []
    for (env, algo, alpha), grp in df_trained.groupby(["env", "algo", "alpha"]):
        mvs_vals = grp["mvs"].values
        rows.append({
            "env": env, "algo": algo, "alpha": alpha,
            "mvs_mean": np.mean(mvs_vals),
            "mvs_std": np.std(mvs_vals, ddof=1) if len(mvs_vals) > 1 else 0.0,
            "mvs_ci95": ci_95(mvs_vals),
            "n_seeds": len(mvs_vals),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "phase1_summary.csv", index=False)
    print(f"Wrote {results_dir / 'phase1_summary.csv'} ({len(summary)} rows)")

    # Monotonicity: Spearman rho of MVS vs alpha per (env, algo)
    mono_rows = []
    for (env, algo), grp in df_trained.groupby(["env", "algo"]):
        rho, pval = stats.spearmanr(grp["alpha"], grp["mvs"])
        mono_rows.append({
            "env": env, "algo": algo,
            "spearman_rho": rho, "spearman_pval": pval,
            "significant": pval < 0.05,
        })
    mono = pd.DataFrame(mono_rows)
    mono.to_csv(results_dir / "phase1_monotonicity.csv", index=False)
    print(f"Wrote {results_dir / 'phase1_monotonicity.csv'} ({len(mono)} rows)")

    # Welch t-test: clean vs alpha=0.9, with BH FDR correction
    welch_rows = []
    for (env, algo), grp in df_trained.groupby(["env", "algo"]):
        clean = grp[grp["alpha"] == 0.0]["mvs"].values
        noised = grp[grp["alpha"] == 0.9]["mvs"].values
        if len(clean) >= 2 and len(noised) >= 2:
            t_stat, p_val = stats.ttest_ind(clean, noised, equal_var=False)
            welch_rows.append({
                "env": env, "algo": algo,
                "clean_mean": np.mean(clean), "noised_mean": np.mean(noised),
                "t_stat": t_stat, "p_val": p_val,
            })
    if welch_rows:
        welch = pd.DataFrame(welch_rows)
        welch["p_val_bh"] = benjamini_hochberg(welch["p_val"].values)
        welch["significant"] = welch["p_val_bh"] < 0.05
        welch.to_csv(results_dir / "phase1_welch.csv", index=False)
        print(f"Wrote {results_dir / 'phase1_welch.csv'} ({len(welch)} rows)")

    # Random-policy specificity
    df_random = df[df["phase"] == "phase1b_random"].copy()
    if not df_random.empty:
        rand_rows = []
        for env, grp in df_random.groupby("env"):
            mvs_vals = grp["mvs"].values
            rand_rows.append({
                "env": env, "algo": "random",
                "mvs_mean": np.mean(mvs_vals),
                "mvs_std": np.std(mvs_vals, ddof=1) if len(mvs_vals) > 1 else 0.0,
                "n_seeds": len(mvs_vals),
            })
        rand_df = pd.DataFrame(rand_rows)
        rand_df.to_csv(results_dir / "phase1_random.csv", index=False)
        print(f"Wrote {results_dir / 'phase1_random.csv'} ({len(rand_df)} rows)")

    return summary


def plot_phase1(summary, results_dir):
    """Plot MVS vs alpha: 2x3 grid, one panel per env."""
    envs = summary["env"].unique()
    n_envs = len(envs)
    nrows = (n_envs + 2) // 3
    ncols = min(n_envs, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, env in enumerate(sorted(envs)):
        ax = axes[idx // ncols][idx % ncols]
        env_data = summary[summary["env"] == env]

        for algo in sorted(env_data["algo"].unique()):
            algo_data = env_data[env_data["algo"] == algo].sort_values("alpha")
            ax.errorbar(algo_data["alpha"], algo_data["mvs_mean"],
                        yerr=algo_data["mvs_ci95"],
                        marker="o", capsize=3, label=algo)

        ax.set_title(env)
        ax.set_xlabel("AR(1) alpha")
        ax.set_ylabel("MVS")
        ax.set_ylim(bottom=-0.02)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_envs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("MVS vs AR(1) Noise Level (Phase 1b)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(results_dir / "phase1_mvs_vs_alpha.pdf", bbox_inches="tight")
    fig.savefig(results_dir / "phase1_mvs_vs_alpha.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote phase1_mvs_vs_alpha.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Phase 2 analysis
# ---------------------------------------------------------------------------

def analyze_phase2(df, results_dir):
    """Produce phase2 summary and degradation CSVs."""
    rows = []
    for (env, algo, alpha), grp in df.groupby(["env", "algo", "alpha"]):
        rewards = grp["mean_reward"].values
        rows.append({
            "env": env, "algo": algo, "alpha": alpha,
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0,
            "reward_ci95": ci_95(rewards),
            "n_seeds": len(rewards),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "phase2_summary.csv", index=False)
    print(f"Wrote {results_dir / 'phase2_summary.csv'} ({len(summary)} rows)")

    # Degradation: ratio vs clean baseline
    deg_rows = []
    for (env, algo), grp in summary.groupby(["env", "algo"]):
        baseline = grp[grp["alpha"] == 0.0]["reward_mean"].values
        if len(baseline) == 0:
            continue
        baseline_val = baseline[0]
        for _, row in grp.iterrows():
            ratio = row["reward_mean"] / baseline_val if abs(baseline_val) > 1e-8 else np.nan
            deg_rows.append({
                "env": env, "algo": algo, "alpha": row["alpha"],
                "reward_mean": row["reward_mean"],
                "baseline_reward": baseline_val,
                "reward_ratio": ratio,
            })
    degradation = pd.DataFrame(deg_rows)
    degradation.to_csv(results_dir / "phase2_degradation.csv", index=False)
    print(f"Wrote {results_dir / 'phase2_degradation.csv'} ({len(degradation)} rows)")

    # Welch t-test: clean vs alpha=0.9 reward, with BH FDR correction
    welch_rows = []
    for (env, algo), grp in df.groupby(["env", "algo"]):
        clean = grp[grp["alpha"] == 0.0]["mean_reward"].values
        noised = grp[grp["alpha"] == 0.9]["mean_reward"].values
        if len(clean) >= 2 and len(noised) >= 2:
            t_stat, p_val = stats.ttest_ind(clean, noised, equal_var=False)
            welch_rows.append({
                "env": env, "algo": algo,
                "clean_mean": np.mean(clean), "noised_mean": np.mean(noised),
                "t_stat": t_stat, "p_val": p_val,
            })
    if welch_rows:
        welch = pd.DataFrame(welch_rows)
        welch["p_val_bh"] = benjamini_hochberg(welch["p_val"].values)
        welch["significant"] = welch["p_val_bh"] < 0.05
        welch.to_csv(results_dir / "phase2_welch.csv", index=False)
        print(f"Wrote {results_dir / 'phase2_welch.csv'} ({len(welch)} rows)")

    return summary, degradation


def plot_phase2(summary, results_dir):
    """Plot reward vs alpha: 2x3 grid."""
    envs = summary["env"].unique()
    n_envs = len(envs)
    nrows = (n_envs + 2) // 3
    ncols = min(n_envs, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, env in enumerate(sorted(envs)):
        ax = axes[idx // ncols][idx % ncols]
        env_data = summary[summary["env"] == env]

        for algo in sorted(env_data["algo"].unique()):
            algo_data = env_data[env_data["algo"] == algo].sort_values("alpha")
            ax.errorbar(algo_data["alpha"], algo_data["reward_mean"],
                        yerr=algo_data["reward_ci95"],
                        marker="s", capsize=3, label=algo)

        ax.set_title(env)
        ax.set_xlabel("AR(1) alpha")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n_envs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Reward vs AR(1) Noise Level (Phase 2)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(results_dir / "phase2_reward_vs_alpha.pdf", bbox_inches="tight")
    fig.savefig(results_dir / "phase2_reward_vs_alpha.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote phase2_reward_vs_alpha.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def analyze_combined(p1_summary, p2_degradation, results_dir, plots_dir):
    """Scatter: MVS vs reward degradation; Spearman correlation."""
    # Merge on (env, algo, alpha)
    merged = pd.merge(p1_summary, p2_degradation,
                       on=["env", "algo", "alpha"], how="inner")

    if merged.empty:
        print("No overlapping (env, algo, alpha) between phase1 and phase2.")
        return

    # Filter out alpha=0 (baseline, ratio=1.0 by definition)
    merged_noised = merged[merged["alpha"] > 0].copy()

    if len(merged_noised) < 3:
        print("Too few data points for combined analysis.")
        return

    # Spearman: MVS vs reward_ratio
    rho, pval = stats.spearmanr(merged_noised["mvs_mean"],
                                 merged_noised["reward_ratio"])
    print(f"\nCombined: Spearman rho(MVS, reward_ratio) = {rho:.3f}, p = {pval:.4f}")

    combined_stats = pd.DataFrame([{
        "spearman_rho": rho, "spearman_pval": pval,
        "n_points": len(merged_noised),
        "significant": pval < 0.05,
    }])
    combined_stats.to_csv(results_dir / "combined_correlation.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    envs = merged_noised["env"].unique()
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, env in enumerate(sorted(envs)):
        env_data = merged_noised[merged_noised["env"] == env]
        ax.scatter(env_data["mvs_mean"], env_data["reward_ratio"],
                   marker=markers[i % len(markers)], s=60, label=env, alpha=0.8)

    ax.set_xlabel("MVS (Phase 1b)")
    ax.set_ylabel("Reward Ratio (vs clean baseline)")
    ax.set_title(f"MVS vs Reward Degradation\n"
                 f"Spearman rho={rho:.3f}, p={pval:.4f}")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "combined_mvs_vs_reward.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / "combined_mvs_vs_reward.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote combined_mvs_vs_reward.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=str, default="results/final")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1
    phase1_path = results_dir / "phase1_mvs.jsonl"
    if phase1_path.exists():
        print("=== Phase 1 Analysis ===")
        df1 = load_phase1(results_dir)
        if not df1.empty:
            p1_summary = analyze_phase1(df1, results_dir)
            if not p1_summary.empty:
                plot_phase1(p1_summary, plots_dir)
        else:
            print("No successful phase1 results.")
            p1_summary = pd.DataFrame()
    else:
        print(f"No phase1 results at {phase1_path}")
        p1_summary = pd.DataFrame()

    # Phase 2
    phase2_path = results_dir / "phase2_reward.jsonl"
    if phase2_path.exists():
        print("\n=== Phase 2 Analysis ===")
        df2 = load_phase2(results_dir)
        if not df2.empty:
            p2_summary, p2_degradation = analyze_phase2(df2, results_dir)
            if not p2_summary.empty:
                plot_phase2(p2_summary, plots_dir)
        else:
            print("No successful phase2 results.")
            p2_degradation = pd.DataFrame()
    else:
        print(f"No phase2 results at {phase2_path}")
        p2_degradation = pd.DataFrame()

    # Combined
    if not p1_summary.empty and not p2_degradation.empty:
        print("\n=== Combined Analysis ===")
        analyze_combined(p1_summary, p2_degradation, results_dir, plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
