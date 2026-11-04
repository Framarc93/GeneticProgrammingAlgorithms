"""Collect completed regression experiments and plot their aggregate results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def collect_results(batch_directory):
    """Aggregate manifest-listed runs; carry terminal fitness forward after early stopping.

    The band is sample standard deviation (ddof=1), not a confidence interval.
    Each boxplot uses scores of the same test-selected champion within each run.
    """
    root = Path(batch_directory)
    manifest = json.loads((root / "batch.json").read_text())
    records = [json.loads((root / run / "result.json").read_text()) for run in manifest["runs"]]
    if not records:
        raise ValueError("No completed experiments to aggregate")
    curves = [np.asarray(record["training_history"], dtype=float) for record in records]
    if any(curve.size == 0 or not np.isfinite(curve).all() for curve in curves):
        raise ValueError("Training histories must be nonempty and finite")
    count = max(map(len, curves))
    aligned = np.array([np.pad(curve, (0, count - len(curve)), mode="edge") for curve in curves])
    mean = aligned.mean(axis=0)
    std = aligned.std(axis=0, ddof=1) if len(records) > 1 else np.zeros(count)
    evolution = pd.DataFrame(
        {
            "generation": np.arange(count),
            "mean": mean,
            "std": std,
            "active_runs": [sum(len(c) > i for c in curves) for i in range(count)],
        }
    )
    evolution.to_csv(root / "training_evolution.csv", index=False)
    columns = [
        "experiment",
        "seed",
        "selected_generation",
        "train_rmse",
        "validation_rmse",
        "test_rmse",
        "nodes",
        "evolution_seconds",
    ]
    table = pd.DataFrame(records)[columns]
    table.to_csv(root / "experiment_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for curve in aligned:
        ax.plot(np.arange(count), curve, color="#4361ee", alpha=0.25, linewidth=1)
    ax.plot(np.arange(count), mean, color="#2536a8", linewidth=2, label="Mean generation-best RMSE")
    ax.fill_between(
        np.arange(count),
        mean - std,
        mean + std,
        color="#4361ee",
        alpha=0.2,
        label="±1 sample standard deviation",
    )
    ax.set(
        xlabel="Generation (0 = initial population)",
        ylabel="Training RMSE (scaled target)",
        title=f"Training evolution — {len(records)} experiments",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(root / "training_evolution.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    scores = [table[column].to_numpy() for column in columns[3:6]]
    ax.boxplot(scores, tick_labels=["Training", "Validation", "Test"])
    ax.set(
        ylabel="RMSE (scaled target)",
        title=f"Test-selected champion per run — {len(records)} experiments",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(root / "best_fitness_boxplots.png", dpi=160)
    plt.close(fig)
    for column, xlabel, title, filename in (
        (
            "nodes",
            "Model complexity (total GP tree nodes)",
            "Model complexity versus fitness",
            "complexity_vs_fitness.png",
        ),
        (
            "evolution_seconds",
            "Evolution runtime (seconds)",
            "Runtime versus fitness",
            "runtime_vs_fitness.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        ax.scatter(table[column], table["test_rmse"], color="#4361ee", s=65, zorder=3)
        for row in table.itertuples():
            ax.annotate(
                f"Run {row.experiment + 1}",
                (getattr(row, column), row.test_rmse),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=9,
            )
        ax.set(
            xlabel=xlabel,
            ylabel="Test RMSE (scaled target; lower is better)",
            title=f"{title} — selected champion per run",
        )
        if column == "nodes":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=0.2, y=0.2)
        ax.grid(alpha=0.2)
        fig.savefig(root / filename, dpi=160)
        plt.close(fig)
    return table, evolution
