"""Quick analysis of ``run_scaling.py`` output: N_all(eps) per n and method, slope fits, one plot.

For every (method, n, instance) the budget configurations (plugin: m; robust_q_ftrl: (N, K))
whose share of sampling seeds with ``gap_out <= eps`` is at least ``success_rate`` count as
successful; N_all is the smallest sample count among them (empty if none succeeds:
"censored", reported separately). Several CSVs can be passed (e.g. a plug-in sweep and a
Robust-Q-FTRL sweep of the same family and settings); rows are simply concatenated.

Per (method, n) the summary also reports ``lowest_budget``: the cheapest budget configuration
whose share of ``gap_out <= eps`` over all runs (all instances and seeds together) is at least
``success_rate``, and its sample count ``N_lowest`` (empty if no configuration succeeds).
``N_lowest_lo`` / ``N_lowest_hi`` give a 95 % bootstrap confidence interval for ``N_lowest``:
the instances (each with all its seeds) are resampled with replacement and the lowest budget is
recomputed (``--boot`` resamples, fixed seed). An upper end of ``inf`` means that in more than
2.5 % of the resamples no budget of the grid succeeds. The plot shows ``N_lowest`` per n with
this interval (log scale on the y axis only).

Fits, on the geometric mean over instances:

- semi-log: ``log2 N_all = a + b n``; b = 1 means a factor 2 = A per extra agent (exponential).
- log-log:  ``log N_all = a + b log n``; b = 1 means linear growth in n (polynomial).

The real statistics (confidence intervals, report plots) are Tiago's part; this is a check.

    python experiments/analyze_scaling.py results/scaling_n/<ts>/runs.csv results/scaling_n_f1_rq/<ts>/runs.csv --plot out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BUDGET = {"plugin": ["m"], "robust_q_ftrl": ["N", "K"]}


def n_all_table(df: pd.DataFrame, eps: float, success_rate: float) -> pd.DataFrame:
    """One row per (method, n, instance_seed) with N_all (NaN if no budget succeeds)."""
    rows = []
    for method, keys in BUDGET.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        share = (
            sub.assign(ok=sub["gap_out"] <= eps)
            .groupby(["n", "instance_seed", *keys])
            .agg(share=("ok", "mean"), n_samples=("n_samples", "first"))
            .reset_index()
        )
        for (n, inst), g in share.groupby(["n", "instance_seed"]):
            good = g[g["share"] >= success_rate - 1e-9]
            best = good.loc[good["n_samples"].idxmin()] if len(good) else None
            rows.append({
                "method": method, "n": n, "instance_seed": inst,
                "N_all": float(best["n_samples"]) if best is not None else np.nan,
                "budget": ", ".join(f"{k}={int(best[k])}" for k in keys) if best is not None else "",
                # K = 1 learns nothing (output = uniform policy): such an instance is trivial.
                "trivial": best is not None and "K" in keys and int(best["K"]) == 1,
            })
    return pd.DataFrame(rows)


def _lowest_budget_ci(
    g: pd.DataFrame, keys: list[str], success_rate: float, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """95 % bootstrap interval of the lowest successful budget, resampling instances.

    ``g`` holds the runs of one (method, n) with column ``ok``.
    """
    counts = g.groupby(["instance_seed", *keys]).agg(ok=("ok", "sum"), tot=("ok", "size"), n_samples=("n_samples", "first"))
    ok = counts["ok"].unstack(keys).fillna(0.0)              # (instances, budgets)
    tot = counts["tot"].unstack(keys).fillna(0.0)
    cost = counts["n_samples"].groupby(keys).first().reindex(ok.columns).to_numpy()
    n_inst = len(ok)
    W = rng.multinomial(n_inst, np.full(n_inst, 1.0 / n_inst), size=n_boot)   # (n_boot, instances)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = (W @ ok.to_numpy()) / (W @ tot.to_numpy())                      # (n_boot, budgets)
    good = share >= success_rate - 1e-9
    lowest = np.where(good, cost[None, :], np.inf).min(axis=1)
    lo, hi = np.percentile(lowest, [2.5, 97.5])
    return float(lo), float(hi)


def lowest_budget_table(
    df: pd.DataFrame, eps: float, success_rate: float, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """One row per (method, n): the cheapest budget whose pooled success share is high enough.

    The share is taken over all runs of that budget (every instance and seed). Also returns a
    95 % bootstrap interval for its sample count (see the module docstring).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for method, keys in BUDGET.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        share = (
            sub.assign(ok=sub["gap_out"] <= eps)
            .groupby(["n", *keys])
            .agg(share=("ok", "mean"), n_samples=("n_samples", "first"))
            .reset_index()
        )
        runs = sub.assign(ok=sub["gap_out"] <= eps)
        for n, g in share.groupby("n"):
            good = g[g["share"] >= success_rate - 1e-9]
            best = good.loc[good["n_samples"].idxmin()] if len(good) else None
            lo, hi = _lowest_budget_ci(runs[runs["n"] == n], keys, success_rate, n_boot, rng)
            rows.append({
                "method": method, "n": n,
                "lowest_budget": ", ".join(f"{k}={int(best[k])}" for k in keys) if best is not None else "",
                "N_lowest": float(best["n_samples"]) if best is not None else np.nan,
                "N_lowest_lo": lo, "N_lowest_hi": hi,
            })
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, lowest: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, n), g in table.groupby(["method", "n"]):
        vals = g["N_all"].dropna()
        out.append({
            "method": method, "n": n, "solved": f"{len(vals)}/{len(g)}",
            "geo_mean": float(np.exp(np.log(vals).mean())) if len(vals) else np.nan,
            "median": float(vals.median()) if len(vals) else np.nan,
            "min": float(vals.min()) if len(vals) else np.nan,
            "max": float(vals.max()) if len(vals) else np.nan,
            "solved_by_K1": int(g["trivial"].sum()) if method == "robust_q_ftrl" else "",
        })
    return pd.DataFrame(out).merge(lowest, on=["method", "n"], how="left")


def fit(summary: pd.DataFrame, method: str) -> dict:
    s = summary[(summary["method"] == method)].dropna(subset=["geo_mean"])
    n, y = s["n"].to_numpy(float), s["geo_mean"].to_numpy()
    if len(n) < 2:
        return {}
    semi = np.polyfit(n, np.log2(y), 1)[0]
    loglog = np.polyfit(np.log(n), np.log(y), 1)[0]
    return {"semilog_slope_log2_per_agent": semi, "factor_per_agent": 2**semi, "loglog_slope": loglog}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--success-rate", type=float, default=0.9)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the CI of N_lowest")
    args = ap.parse_args(argv)

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    table = n_all_table(df, args.eps, args.success_rate)
    summary = summarize(table, lowest_budget_table(df, args.eps, args.success_rate, args.boot))
    pd.set_option("display.width", 160)
    print(f"N_all(eps = {args.eps}), success rate {args.success_rate}:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    for method in BUDGET:
        f = fit(summary, method)
        if f:
            print(f"{method:<14} semi-log slope {f['semilog_slope_log2_per_agent']:.2f} "
                  f"(x{f['factor_per_agent']:.2f} per agent), log-log slope {f['loglog_slope']:.2f}")

    if "exact" in set(df["method"]):
        floor = df[df["method"] == "exact"].groupby("n")["gap_out"].agg(["mean", "max"])
        print("\nSolver floor (exact, gap_out):")
        print(floor.T.to_string(float_format=lambda x: f"{x:.3f}"))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.2))
        styles = {"plugin": ("tab:red", "o", "plug-in"), "robust_q_ftrl": ("tab:blue", "s", "Robust-Q-FTRL")}
        for method, (color, marker, label) in styles.items():
            s = summary[summary["method"] == method].dropna(subset=["N_lowest"])
            if s.empty:
                continue
            y = s["N_lowest"].to_numpy()
            hi = s["N_lowest_hi"].to_numpy()
            open_top = ~np.isfinite(hi)                   # CI open upwards: no budget in the grid
            hi = np.where(open_top, y, hi)
            yerr = np.vstack([y - s["N_lowest_lo"].to_numpy(), hi - y])
            ax.errorbar(s["n"], y, yerr=yerr, color=color, marker=marker, capsize=4,
                        label=f"{label} (95 % CI)")
            if open_top.any():
                ax.scatter(s["n"][open_top], y[open_top] * 1.6, color=color, marker="^", s=30)
        ax.set_yscale("log")
        ax.set_xticks(sorted(df["n"].unique()))
        ax.set_xlabel("number of agents n")
        ax.set_ylabel(r"lowest budget $N_{\mathrm{lowest}}(\varepsilon)$ (samples)")
        ax.set_title(f"eps = {args.eps}, success rate {args.success_rate}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot: {args.plot}")


if __name__ == "__main__":
    main()
