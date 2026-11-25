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
2.5 % of the resamples no budget of the grid succeeds.

Every statistic over the games is the **geometric mean** (tables, fits and plot points), because
N_all spans orders of magnitude and the fits are done on log N_all.

The plot (``--plot``) shows the geometric mean of N_all over the games per n with a 95 %
bootstrap interval (games resampled), each method's growth model fitted on the informative n
(dashed) and the region where F1 is (partly) trivial (shaded; see ``plot``). Log scale on the y
axis only, so an exponential is a straight line and a power law n^c bends over.

Fits, on the geometric mean over instances, by default only on the informative n (below the
first n where any instance is solved with K = 1; ``--fit-n`` overrides):

- semi-log (exponential): ``log2 N_all = a + b n``; 2^b is the factor per extra agent
  (2 = A: the 2^n of a joint-action method). Drawn for plug-in.
- log-log (power law):  ``log N_all = a + c log n``; c = 1 means linear growth in n. Drawn for
  Robust-Q-FTRL.

Both are printed for both methods.

The real statistics (confidence intervals, report plots) are Tiago's part; this is a check.

    python experiments/analyze_scaling.py results/scaling_n/<ts>/runs.csv --methods plugin --plot out.png
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
        hsa = sub.groupby("n")[["H", "S", "A"]].first()
        unit = hsa["H"] * hsa["S"] * hsa["A"]                     # H·S·A per n
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
                # Samples per agent and own action, the same normalisation for every method:
                # Robust-Q-FTRL (H·S·n·A·K·N) gives K·N, plug-in (H·S·Aⁿ·m) gives A^(n-1)·m / n.
                "per_agent": float(best["n_samples"]) / (unit.loc[n] * n) if best is not None else np.nan,
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


# Growth model each method is expected to follow, and the one drawn in the plot.
MODEL = {"plugin": "exp", "robust_q_ftrl": "power"}


def fit(summary: pd.DataFrame, method: str, fit_n: list[int] | None = None) -> dict:
    """Semi-log and log-log fits of the geometric mean of N_all over n.

    ``fit_n``: the n to fit on (default: every n with a geometric mean).
    """
    s = summary[(summary["method"] == method)].dropna(subset=["geo_mean"])
    if fit_n is not None:
        s = s[s["n"].isin(fit_n)]
    n, y = s["n"].to_numpy(float), s["geo_mean"].to_numpy()
    if len(n) < 2:
        return {}
    semi, semi_icpt = np.polyfit(n, np.log2(y), 1)
    loglog, loglog_icpt = np.polyfit(np.log(n), np.log(y), 1)
    return {
        "n": [int(v) for v in n],
        "semilog_slope_log2_per_agent": semi, "factor_per_agent": 2**semi, "semilog_intercept_log2": semi_icpt,
        "loglog_slope": loglog, "loglog_intercept": loglog_icpt,
    }


def fit_curve(f: dict, model: str, x: np.ndarray) -> np.ndarray:
    """Evaluate a fit from ``fit`` at ``x`` under ``model`` ("exp" or "power")."""
    if model == "exp":
        return 2.0 ** (f["semilog_intercept_log2"] + f["semilog_slope_log2_per_agent"] * x)
    return np.exp(f["loglog_intercept"] + f["loglog_slope"] * np.log(x))


def informative_n(summary: pd.DataFrame, start: float | None) -> list[int]:
    """The n below the trivial region (all n if there is none)."""
    ns = sorted(int(v) for v in summary["n"].unique())
    return [n for n in ns if start is None or n < start]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

INK = "#21363a"          # lines, markers, text
FIT = "#6e7180"          # dashed fit
SHADE = "#f8eceb"        # trivial region
SHADE_TEXT = "#b23a33"
BACKGROUND = "#f9f9f9"
GRID = "#e4e4e4"
COLORS = {"plugin": INK, "robust_q_ftrl": "#3a6ea5"}
LABELS = {"plugin": "plug-in", "robust_q_ftrl": "Robust-Q-FTRL"}


def mean_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean over instances and its 95 % bootstrap interval (instances resampled)."""
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def geo_mean_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Geometric mean over instances and its 95 % bootstrap interval (instances resampled)."""
    logs = np.log(values)
    boot = rng.choice(logs, size=(n_boot, len(logs)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.exp(logs.mean())), float(np.exp(lo)), float(np.exp(hi))


def trivial_start(summary: pd.DataFrame) -> float | None:
    """First n at which any instance is solved by Robust-Q-FTRL with K = 1 (uniform policy).

    Below it is the informative range used in the docs; from it on N_all partly measures the game, not learning.

    Needs Robust-Q-FTRL runs with K = 1; returns None otherwise.
    """
    s = summary[summary["method"] == "robust_q_ftrl"]
    if s.empty or (s["solved_by_K1"] == 0).all():
        return None
    trivial = s.loc[s["solved_by_K1"].astype(int) > 0, "n"]
    return float(trivial.min()) if len(trivial) else None


def plot(
    table: pd.DataFrame, summary: pd.DataFrame, eps: float, path: Path, methods: list[str] | None = None,
    fit_n: list[int] | None = None, trivial_from: float | None = None, n_boot: int = 2000, seed: int = 0,
) -> None:
    """Geometric mean N_all over the games with a 95 % bootstrap CI, model fits and the trivial region.

    - The fit (dashed) is each method's growth model from ``MODEL`` (plug-in: exponential
      ``∝ bⁿ``, a straight line on the log axis; Robust-Q-FTRL: power law ``∝ n^c``, which bends
      over), fitted on the geometric means at the n in ``fit_n``; by default all n below the
      trivial region. The same fits are printed by ``main``. Drawn over the whole range of n;
      inside the trivial region it is an extrapolation.
    - The trivial region (shaded) starts at ``trivial_from``; by default at the first n where
      any instance is solved with K = 1 (see ``trivial_start``), none if that is unknown.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    rng = np.random.default_rng(seed)
    methods = [m for m in (methods or BUDGET) if m in set(table["method"])]
    ns = sorted(table["n"].unique())
    start = trivial_from if trivial_from is not None else trivial_start(summary)

    single = len(methods) == 1
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 15, "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(7.75, 5.9 if single else 6.6))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    if start is not None:
        ax.axvspan(start - 0.5, ns[-1] + 0.4, color=SHADE, zorder=0, lw=0)
        # Above the axes, so it never collides with the curves or fits inside the region.
        narrow = ns[-1] + 0.4 - (start - 0.5) < 2.5
        ax.text((start - 0.5 + ns[-1] + 0.4) / 2, 1.01, "F1 (partly) trivial", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=SHADE_TEXT, fontsize=12 if narrow else 14, zorder=1)

    for method in methods:
        color = COLORS.get(method, INK)
        prefix = f"{LABELS.get(method, method)}: " if len(methods) > 1 else ""
        rows = []
        for n, g in table[table["method"] == method].groupby("n"):
            vals = g["N_all"].dropna().to_numpy()
            if len(vals):
                rows.append((n, len(vals), *geo_mean_ci(vals, n_boot, rng)))
        if not rows:
            continue
        n_arr, counts, gmean, lo, hi = map(np.array, zip(*rows))
        n_games = int(counts.max())
        ax.errorbar(n_arr, gmean, yerr=[gmean - lo, hi - gmean], color=color, lw=2.2, marker="o", ms=11,
                    capsize=5, elinewidth=1.3, capthick=1.3, zorder=3,
                    label=f"{prefix}geo. mean of {n_games} games, 95 % CI")

        f = fit(summary, method, fit_n if fit_n else informative_n(summary, start))
        if f:
            model = MODEL.get(method, "exp")
            x = np.linspace(ns[0] - 0.2, ns[-1] + 0.15, 100)
            used = f["n"]
            fit_range = (f"{used[0]}–{used[-1]}" if np.all(np.diff(used) == 1)
                         else ", ".join(map(str, used)))
            law = (rf"$\propto {f['factor_per_agent']:.2f}^{{\,n}}$" if model == "exp"
                   else rf"$\propto n^{{{f['loglog_slope']:.2f}}}$")
            ax.plot(x, fit_curve(f, model, x), ls=(0, (5, 4)), color=FIT if len(methods) == 1 else color,
                    lw=1.6, alpha=0.9 if len(methods) == 1 else 0.75, zorder=2,
                    label=f"{prefix}fit {law}, $n$ = {fit_range}")

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(ns)
    ax.set_xlim(ns[0] - 0.4, ns[-1] + 0.4)
    ax.set_xlabel("number of agents $n$", labelpad=10, fontsize=16)
    ax.set_ylabel(rf"$N_{{\mathrm{{all}}}}(\varepsilon = {eps:g})$", labelpad=8, fontsize=17)
    ax.grid(True, which="major", color=GRID, lw=1.0, zorder=1)
    ax.tick_params(colors=INK, which="both", length=0, pad=8)
    ax.tick_params(axis="x", length=8, color=GRID, direction="inout")
    for side, spine in ax.spines.items():
        spine.set_color(INK if side in ("left", "bottom") else GRID)
        spine.set_linewidth(1.2 if side in ("left", "bottom") else 0.8)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(INK)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: ("fit " in labels[i], i))   # means first, then fits
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    if single:
        ax.legend(handles, labels, loc="upper left", frameon=False, labelcolor=INK, handlelength=3.2,
                  borderaxespad=1.0)
    else:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
                  labelcolor=INK, handlelength=2.6, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


UNIT_LABEL = {
    "plugin": r"plug-in $= A^{n-1} m\,/\,n$",
    "robust_q_ftrl": r"Robust-Q-FTRL $= K \cdot N$",
}


def plot_budget(
    table: pd.DataFrame, summary: pd.DataFrame, eps: float, path: Path, methods: list[str] | None = None,
    trivial_from: float | None = None, n_boot: int = 2000, seed: int = 0,
) -> None:
    """Samples per agent and own action, ``N_all / (H·S·A·n)``, against n, for every method.

    Every method is divided by the same factor, so a flat line means the total cost is linear
    in n and a rising line means faster than linear. Robust-Q-FTRL (``N_all = H·S·n·A·K·N``)
    gives exactly K·N; plug-in (``N_all = H·S·Aⁿ·m``) gives ``A^(n-1)·m / n``, which rises even
    at constant m. Points: geometric mean over the games with a 95 % bootstrap CI; small dots:
    the individual games. Log₂ y axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    rng = np.random.default_rng(seed)
    methods = [m for m in (methods or BUDGET) if m in set(table["method"])]
    ns = sorted(table["n"].unique())
    start = trivial_from if trivial_from is not None else trivial_start(summary)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 15, "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(7.75, 6.6))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    if start is not None:
        ax.axvspan(start - 0.5, ns[-1] + 0.4, color=SHADE, zorder=0, lw=0)
        narrow = ns[-1] + 0.4 - (start - 0.5) < 2.5
        ax.text((start - 0.5 + ns[-1] + 0.4) / 2, 1.01, "F1 (partly) trivial", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=SHADE_TEXT, fontsize=12 if narrow else 14, zorder=1)

    offsets = {m: (i - (len(methods) - 1) / 2) * 0.12 for i, m in enumerate(methods)}
    all_vals = []
    for method in methods:
        color = COLORS.get(method, INK)
        rows = []
        for n, g in table[table["method"] == method].groupby("n"):
            vals = g["per_agent"].dropna().to_numpy()
            if len(vals):
                all_vals.extend(vals)
                jitter = rng.uniform(-0.07, 0.07, len(vals))
                ax.scatter(n + offsets[method] + jitter, vals, s=14, color=color, alpha=0.25, lw=0, zorder=2)
                rows.append((n, *geo_mean_ci(vals, n_boot, rng)))
        if not rows:
            continue
        n_arr, gmean, lo, hi = map(np.array, zip(*rows))
        ax.errorbar(n_arr + offsets[method], gmean, yerr=[gmean - lo, hi - gmean], color=color, lw=2.2, marker="o",
                    ms=10, capsize=5, elinewidth=1.3, capthick=1.3, zorder=3, label=UNIT_LABEL.get(method, method))

    ax.set_yscale("log", base=2)
    bot = int(np.floor(np.log2(min(all_vals)))) if all_vals else 0
    top = int(np.ceil(np.log2(max(all_vals)))) if all_vals else 1
    ks = list(range(bot, top + 1))
    ax.yaxis.set_major_locator(FixedLocator([2.0**k for k in ks]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{2.0**k:g}" for k in ks])
    ax.set_ylim(2.0 ** (bot - 0.5), 2.0 ** (top + 0.5))
    ax.set_xticks(ns)
    ax.set_xlim(ns[0] - 0.4, ns[-1] + 0.4)
    ax.set_xlabel("number of agents $n$", labelpad=10, fontsize=16)
    ax.set_ylabel(rf"$N_{{\mathrm{{all}}}}(\varepsilon = {eps:g})\,/\,(H S A\, n)$", labelpad=8, fontsize=17)
    ax.grid(True, which="major", color=GRID, lw=1.0, zorder=1)
    ax.tick_params(colors=INK, which="both", length=0, pad=8)
    ax.tick_params(axis="x", length=8, color=GRID, direction="inout")
    for side, spine in ax.spines.items():
        spine.set_color(INK if side in ("left", "bottom") else GRID)
        spine.set_linewidth(1.2 if side in ("left", "bottom") else 0.8)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, labelcolor=INK,
              handlelength=2.6, fontsize=13, columnspacing=2.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--success-rate", type=float, default=0.9)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--budget-plot", type=Path, default=None,
                    help="plot N_all / (H·S·A·n), samples per agent and own action, per n; flat = linear in n")
    ap.add_argument("--methods", nargs="+", choices=list(BUDGET), help="methods to plot (default: all in the CSVs)")
    ap.add_argument("--fit-n", nargs="+", type=int, help="n used for the dashed fit (default: all n below the trivial region)")
    ap.add_argument("--trivial-from", type=float,
                    help="first n of the shaded trivial region (default: first n where any instance is solved with K = 1)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the CI of N_lowest")
    args = ap.parse_args(argv)

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    table = n_all_table(df, args.eps, args.success_rate)
    summary = summarize(table, lowest_budget_table(df, args.eps, args.success_rate, args.boot))
    pd.set_option("display.width", 160)
    print(f"N_all(eps = {args.eps}), success rate {args.success_rate}:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    start = args.trivial_from if args.trivial_from is not None else trivial_start(summary)
    fit_ns = args.fit_n if args.fit_n else informative_n(summary, start)
    print(f"\nFits on the geometric means, n = {fit_ns} "
          f"({'--fit-n' if args.fit_n else 'informative range'}):")
    for method in BUDGET:
        f = fit(summary, method, fit_ns)
        if f:
            print(f"{method:<14} semi-log slope {f['semilog_slope_log2_per_agent']:.2f} "
                  f"(x{f['factor_per_agent']:.2f} per agent), log-log slope {f['loglog_slope']:.2f}"
                  f"   [plotted: {'exponential' if MODEL[method] == 'exp' else 'power law'}]")

    if "exact" in set(df["method"]):
        floor = df[df["method"] == "exact"].groupby("n")["gap_out"].agg(["mean", "max"])
        print("\nSolver floor (exact, gap_out):")
        print(floor.T.to_string(float_format=lambda x: f"{x:.3f}"))

    if args.plot:
        plot(table, summary, args.eps, args.plot, methods=args.methods, fit_n=args.fit_n,
             trivial_from=args.trivial_from, n_boot=args.boot)
        print(f"\nplot: {args.plot}")
    if args.budget_plot:
        plot_budget(table, summary, args.eps, args.budget_plot, methods=args.methods,
                    trivial_from=args.trivial_from, n_boot=args.boot)
        print(f"budget plot: {args.budget_plot}")


if __name__ == "__main__":
    main()
