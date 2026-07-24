from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_bars(metrics: pd.DataFrame, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ax = metrics.pivot(index="domain_set", columns="metric", values="estimate").plot(kind="bar")
    ax.set_xlabel("Domain set")
    ax.set_ylabel("Estimate")
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
