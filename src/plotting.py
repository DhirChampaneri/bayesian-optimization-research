# src/plotting.py

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(equity_dict: dict[str, pd.Series]) -> None:
    """
    Plot multiple equity curves on one chart.
    
    equity_dict: { "label": equity_series }
    """
    plt.figure(figsize=(10, 5))

    for label, equity in equity_dict.items():
        equity.plot(label=label)

    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Equity Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spread_and_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    start: str | None = None,
    end: str | None = None,
) -> None:
    """
    Plot the spread and z-score with entry/exit thresholds.
    """

    # Optional window slicing
    if start or end:
        spread = spread.loc[start:end]
        zscore = zscore.loc[start:end]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Spread
    axes[0].plot(spread.index, spread.values)
    axes[0].set_ylabel("Spread")
    axes[0].set_title("Spread")

    # Z-score
    axes[1].plot(zscore.index, zscore.values)
    axes[1].axhline(entry_z, linestyle="--")
    axes[1].axhline(-entry_z, linestyle="--")
    axes[1].axhline(exit_z, linestyle=":")
    axes[1].axhline(-exit_z, linestyle=":")
    axes[1].set_ylabel("Z-score")
    axes[1].set_title("Z-score with Trading Thresholds")

    plt.tight_layout()
    plt.show()
