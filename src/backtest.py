# src/backtest.py

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import download_price_data
from .cointegration import get_baseline_weights, construct_spread


def compute_zscore(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Compute rolling z-score of a spread series.

    z_t = (spread_t - mean_{t-lookback:t}) / std_{t-lookback:t}
    """
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std = spread.rolling(window=lookback).std()

    z = (spread - rolling_mean) / rolling_std
    return z


def backtest_mean_reversion(
    spread: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.0,
    lookback: int = 60,
    initial_capital: float = 100_000.0,
) -> dict:
    """
    Simple mean-reversion backtest on a spread.

    Rules (daily):
      - If flat and z >= entry_z  -> short spread (-1)
      - If flat and z <= -entry_z -> long spread (+1)
      - If in a trade and |z| <= exit_z -> close (go flat)
      - If in a trade and |z| >= stop_z -> stop out (go flat)

    We assume PnL from 1 unit of spread = position * change in spread.
    """
    # Compute z-score
    z = compute_zscore(spread, lookback=lookback)
    z = z.dropna()

    # Align spread with z
    spread = spread.loc[z.index]

    # Prepare containers
    idx = spread.index
    position = pd.Series(0.0, index=idx)  # +1 long spread, -1 short, 0 flat

    # Generate position series
    for i in range(1, len(idx)):
        prev_pos = position.iloc[i - 1]
        z_prev = z.iloc[i - 1]

        if prev_pos == 0:
            # We are flat: check for entry
            if z_prev >= entry_z:
                position.iloc[i] = -1.0  # short spread
            elif z_prev <= -entry_z:
                position.iloc[i] = 1.0   # long spread
            else:
                position.iloc[i] = 0.0
        else:
            # We are in a trade: check for exit or stop
            if abs(z_prev) <= exit_z or abs(z_prev) >= stop_z:
                position.iloc[i] = 0.0
            else:
                position.iloc[i] = prev_pos

    # Compute PnL: position * change in spread
    spread_diff = spread.diff()
    pnl = position.shift(1) * spread_diff  # yesterday's position * today's move
    pnl = pnl.fillna(0.0)

    # Convert to returns relative to some notional
    # Here we just treat 1 unit of spread as "1 unit of capital"
    # and then scale to initial_capital.
    daily_return = pnl  # PnL per 1 unit notional
    equity = initial_capital * (1 + daily_return).cumprod()

    # Performance stats
    total_return = equity.iloc[-1] / initial_capital - 1.0

    # Annualized Sharpe (assuming 252 trading days)
    ret_mean = daily_return.mean()
    ret_std = daily_return.std()
    if ret_std > 0:
        sharpe = (ret_mean / ret_std) * np.sqrt(252)
    else:
        sharpe = np.nan

    # Max drawdown
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min()

    stats = {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "num_trades": int((position.diff().abs() / 2).sum()),  # rough count
    }

    return {
        "equity_curve": equity,
        "position": position,
        "pnl": pnl,
        "zscore": z,
        "stats": stats,
    }


if __name__ == "__main__":
    # End-to-end demo using Johansen baseline weights
    prices = download_price_data(
        tickers=("AAPL", "MSFT", "GOOGL"),
        start="2020-01-01",
        end="2025-12-02",
        interval="1d",
    )

    weights = get_baseline_weights(prices)
    print("[backtest] Johansen weights:")
    print(weights)

    spread = construct_spread(prices, weights)
    results = backtest_mean_reversion(
        spread,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        lookback=60,
        initial_capital=100_000.0,
    )

    print("\n[backtest] Performance stats:")
    for k, v in results["stats"].items():
        print(f"{k:>15}: {v:.4f}")
