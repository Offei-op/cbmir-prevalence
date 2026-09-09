"""Paired comparisons, prevalence changes and figures from query-level results."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from .io import QUERIES, write_json

KEYS = ["model", "pathology", "prevalence", "database_seed", "query_id", "K"]


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    out = np.empty(len(p))
    out[order] = adjusted
    return out


def validate_results(df, config):
    needed = set(KEYS + ["precision", "ndcg", "query_path"])
    if not needed.issubset(df) or df[list(needed)].isna().any().any() or df.duplicated(KEYS).any():
        raise ValueError("Missing fields/values or duplicate query-level observations")
    if (
        not np.isfinite(df[["precision", "ndcg"]]).all().all()
        or not df[["precision", "ndcg"]].ge(0).all().all()
        or not df[["precision", "ndcg"]].le(1 + 1e-7).all().all()
    ):
        raise ValueError("Metrics must be finite in [0, 1]")
    expected = pd.MultiIndex.from_product(
        [
            config["models"],
            list(QUERIES),
            config["prevalences"],
            config["database_seeds"],
            config["ks"],
        ],
        names=["model", "pathology", "prevalence", "database_seed", "K"],
    )
    counts = df.groupby(expected.names).size().reindex(expected)
    if (
        len(counts) != df.groupby(expected.names).ngroups
        or counts.isna().any()
        or not counts.eq(config["queries_per_pathology"]).all()
    ):
        raise ValueError("Incomplete or unexpected result grid")
    identity = df.groupby(["pathology", "query_id"]).query_path.nunique()
    if not identity.eq(1).all():
        raise ValueError("Query IDs do not identify fixed images")
    for pathology, sub in df.groupby("pathology"):
        if (
            sub.query_id.nunique() != config["queries_per_pathology"]
            or sub.query_path.nunique() != config["queries_per_pathology"]
        ):
            raise ValueError(f"Query set changed across conditions: {pathology}")


def paired_summary(a, b, samples=10000, seed=42):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size < 2 or not np.isfinite(diff).all():
        raise ValueError("Need at least two complete paired observations")
    if np.all(diff == 0):
        stat, p = 0.0, 1.0
    else:
        stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for start in range(0, samples, 1000):
        n = min(1000, samples - start)
        boot[start : start + n] = rng.choice(diff, (n, len(diff)), replace=True).mean(1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "n_queries": len(diff),
        "mean_difference": diff.mean(),
        "median_difference": np.median(diff),
        "ci_low": low,
        "ci_high": high,
        "wilcoxon_stat": stat,
        "p_raw": p,
    }


def run(config, results_path=None):
    root = Path(config["artifacts_dir"])
    source = (
        Path(results_path)
        if results_path
        else root / "results/query_level_rare_retrieval_results.csv"
    )
    df = pd.read_csv(source)
    validate_results(df, config)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    conditions = ["model", "pathology", "prevalence", "database_seed", "K"]
    agg = df.groupby(conditions, as_index=False).agg(
        precision_mean=("precision", "mean"), ndcg_mean=("ndcg", "mean")
    )
    agg.to_csv(out / "seed_level_results.csv", index=False)
    headline = agg.groupby(["model", "pathology", "prevalence", "K"], as_index=False).agg(
        precision_mean=("precision_mean", "mean"),
        precision_sd=("precision_mean", "std"),
        ndcg_mean=("ndcg_mean", "mean"),
        ndcg_sd=("ndcg_mean", "std"),
    )
    headline["random_precision"] = headline.prevalence
    headline["enrichment"] = headline.precision_mean / headline.prevalence
    headline["excess_precision"] = headline.precision_mean - headline.prevalence
    headline.to_csv(out / "headline.csv", index=False)
    qavg = df.groupby(["model", "pathology", "prevalence", "query_id", "K"], as_index=False)[
        ["precision", "ndcg"]
    ].mean()
    qavg.to_csv(out / "query_seed_averages.csv", index=False)
    primary = qavg[qavg.K == config["primary_k"]]
    omnibus, pairs, changes = [], [], []
    # Inferential tests retain original focus on precision at the configured primary k.
    for (pathology, prevalence), sub in primary.groupby(["pathology", "prevalence"]):
        wide = sub.pivot(index="query_id", columns="model", values="precision").reindex(
            columns=config["models"]
        )
        if wide.isna().any().any():
            raise ValueError("Unpaired model comparisons")
        if len(config["models"]) >= 3:
            if np.all(np.ptp(wide.to_numpy(), axis=1) == 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = friedmanchisquare(*[wide[m] for m in config["models"]])
            omnibus.append(
                {
                    "pathology": pathology,
                    "prevalence": prevalence,
                    "n_queries": len(wide),
                    "chi2": stat,
                    "p_raw": p,
                    "kendalls_W": stat / (len(wide) * (len(wide.columns) - 1)),
                }
            )
        for a, b in config["comparisons"]:
            if a not in wide or b not in wide:
                raise ValueError(f"Comparison requires missing model {a}/{b}")
            pairs.append(
                {
                    "pathology": pathology,
                    "prevalence": prevalence,
                    "model_a": a,
                    "model_b": b,
                    **paired_summary(
                        wide[a], wide[b], config["bootstrap_samples"], config["analysis_seed"]
                    ),
                }
            )
    for (model, pathology), sub in primary.groupby(["model", "pathology"]):
        wide = sub.pivot(index="query_id", columns="prevalence", values="precision")
        hi = max(config["prevalences"])
        for low in sorted(config["prevalences"]):
            if low == hi:
                continue
            changes.append(
                {
                    "model": model,
                    "pathology": pathology,
                    "prevalence_high": hi,
                    "prevalence_low": low,
                    **paired_summary(
                        wide[low], wide[hi], config["bootstrap_samples"], config["analysis_seed"]
                    ),
                }
            )
    for rows, name in [
        (omnibus, "friedman"),
        (pairs, "model_comparisons"),
        (changes, "prevalence_changes"),
    ]:
        frame = pd.DataFrame(rows)
        if len(frame):
            frame["p_holm_global"] = holm(frame.p_raw)
            if name == "model_comparisons":
                frame["p_holm_within_condition"] = frame.groupby(
                    ["pathology", "prevalence"]
                ).p_raw.transform(lambda p: holm(p))
            frame.to_csv(out / f"{name}.csv", index=False)
    # Exploratory complementarity: descriptive oracle, never a deployable model.
    wide = df[df.K == config["primary_k"]].pivot(
        index=["pathology", "prevalence", "database_seed", "query_id"],
        columns="model",
        values="precision",
    )
    winners = wide.eq(wide.max(axis=1), axis=0)
    winners.mean().rename("fraction_tied_for_best").to_csv(out / "exploratory_tied_wins.csv")
    oracle = (
        wide.max(axis=1)
        .groupby(level=["pathology", "prevalence"])
        .mean()
        .rename("oracle_precision")
    )
    best = (
        wide.groupby(level=["pathology", "prevalence"])
        .mean()
        .max(axis=1)
        .rename("best_single_precision")
    )
    comparison = pd.concat([oracle, best], axis=1)
    comparison["oracle_gap"] = comparison.oracle_precision - comparison.best_single_precision
    comparison.to_csv(out / "exploratory_oracle.csv")
    plot_curves(headline, out, config["primary_k"])
    write_json(
        out / "analysis_notes.json",
        {
            "primary_metric": f"Precision@{config['primary_k']}",
            "error_bars": "sample SD of database-seed means, not confidence intervals",
            "tests": "query paired after averaging database seeds; independence across queries is approximate",
            "bootstrap": "pointwise 95% paired query bootstrap; conditional on trained models and sampled galleries",
            "multiplicity": "Holm separately across all model contrasts, prevalence contrasts, and omnibus tests; within-condition values also provided",
            "new_analysis": "prevalence contrasts added during cleanup; do not describe as preregistered",
            "limitations": "Not a crossed query/gallery or training-seed uncertainty model; external patient clustering may be unknown",
        },
    )
    return {"rows": len(df), "output": str(out)}


def plot_curves(headline, out, k):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = headline[headline.K == k]
    for metric in ["precision", "ndcg"]:
        fig, axes = plt.subplots(
            2, 2, figsize=(9, 6), sharex=True, sharey=True, constrained_layout=True
        )
        for ax, target in zip(axes.flat, QUERIES):
            sub = selected[selected.pathology == target]
            for name, model in sub.groupby("model", sort=False):
                model = model.sort_values("prevalence")
                ax.errorbar(
                    model.prevalence * 100,
                    model[f"{metric}_mean"],
                    yerr=model[f"{metric}_sd"].fillna(0),
                    marker="o",
                    markersize=3,
                    capsize=2,
                    label=name,
                )
            if metric == "precision":
                x = np.sort(sub.prevalence.unique())
                ax.plot(x * 100, x, "k--", label="Random")
            ax.set_title(target)
            ax.set_xlabel("Target prevalence (%)")
            ax.set_ylabel(f"{'Precision' if metric == 'precision' else 'nDCG'}@{k}")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=6)
        for extension in ["pdf", "png"]:
            fig.savefig(out / f"{metric}_at{k}.{extension}", dpi=300)
        plt.close(fig)
