"""Quick analysis of ``run_scaling.py`` output: N_all(eps) per n and method, slope fits, one plot.

For every (method, n, instance) the budget configurations (plugin: m; robust_q_ftrl: (N, K))
whose share of sampling seeds with ``gap_out <= eps`` is at least ``success_rate`` count as
successful; N_all is the smallest sample count among them (empty if none succeeds:
"censored", reported separately). Several CSVs can be passed (e.g. a plug-in sweep and a
Robust-Q-FTRL sweep of the same family and settings); rows are simply concatenated.

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
            })
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, n), g in table.groupby(["method", "n"]):
        vals = g["N_all"].dropna()
        out.append({
            "method": method, "n": n, "solved": f"{len(vals)}/{len(g)}",
            "geo_mean": float(np.exp(np.log(vals).mean())) if len(vals) else np.nan,
            "median": float(vals.median()) if len(vals) else np.nan,
            "min": float(vals.min()) if len(vals) else np.nan,
            "max": float(vals.max()) if len(vals) else np.nan,
            "typical_budget": g["budget"].mode().iloc[0] if (g["budget"] != "").any() else "",
        })
    return pd.DataFrame(out)


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
    args = ap.parse_args(argv)

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    table = n_all_table(df, args.eps, args.success_rate)
    summary = summarize(table)
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

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        styles = {"plugin": ("tab:red", "o", "plug-in"), "robust_q_ftrl": ("tab:blue", "s", "Robust-Q-FTRL")}
        for ax, loglog in zip(axes, (False, True)):
            for method, (color, marker, label) in styles.items():
                s = summary[summary["method"] == method]
                t = table[table["method"] == method]
                if s.empty:
                    continue
                ax.scatter(t["n"], t["N_all"], color=color, alpha=0.25, s=12)
                ax.plot(s["n"], s["geo_mean"], color=color, marker=marker, label=label)
            ax.set_yscale("log")
            if loglog:
                ax.set_xscale("log")
                ax.set_xticks(sorted(df["n"].unique()))
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("number of agents n")
            ax.set_ylabel(r"$N_{\mathrm{all}}(\varepsilon)$")
            ax.set_title(("log-log" if loglog else "semi-log") + f", eps = {args.eps}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot: {args.plot}")


if __name__ == "__main__":
    main()
