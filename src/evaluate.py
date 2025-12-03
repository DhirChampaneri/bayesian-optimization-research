# src/evaluate.py

from __future__ import annotations

import pandas as pd

from .data_loader import download_price_data
from .cointegration import get_baseline_weights, construct_spread
from .backtest import backtest_mean_reversion
from .bayes_opt import optimize_weights_with_bo


def rolling_evaluation(
    prices: pd.DataFrame,
    train_window: int = 252 * 2,  # 2 years
    test_window: int = 252,       # 1 year
    step: int = 252,              # slide by 1 year
    n_trials_bo: int = 30,
) -> pd.DataFrame:
    """
    Perform rolling-window evaluation comparing:
      - Johansen baseline
      - BO-optimized weights

    For each window:
      - Train period: find Johansen & BO weights using train data
      - Test period: backtest strategy on test data only

    Returns:
      DataFrame with one row per window and metrics for both methods.
    """
    metrics = []

    idx = prices.index
    n = len(idx)

    start_idx = 0

    while True:
        train_start = start_idx
        train_end = train_start + train_window
        test_end = train_end + test_window

        if test_end > n:
            break

        train_prices = prices.iloc[train_start:train_end]
        test_prices = prices.iloc[train_end:test_end]

        train_start_date = train_prices.index[0]
        train_end_date = train_prices.index[-1]
        test_start_date = test_prices.index[0]
        test_end_date = test_prices.index[-1]

        # Johansen on train
        joh_weights = get_baseline_weights(train_prices)
        joh_spread_test = construct_spread(test_prices, joh_weights)
        joh_results = backtest_mean_reversion(joh_spread_test)

        # BO on train
        _, bo_weights, _ = optimize_weights_with_bo(
            train_prices,
            n_trials=n_trials_bo,
        )
        bo_spread_test = construct_spread(test_prices, bo_weights)
        bo_results = backtest_mean_reversion(bo_spread_test)

        metrics.append({
            "train_start": train_start_date,
            "train_end": train_end_date,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "joh_total_return": joh_results["stats"]["total_return"],
            "joh_sharpe": joh_results["stats"]["sharpe"],
            "joh_max_dd": joh_results["stats"]["max_drawdown"],
            "bo_total_return": bo_results["stats"]["total_return"],
            "bo_sharpe": bo_results["stats"]["sharpe"],
            "bo_max_dd": bo_results["stats"]["max_drawdown"],
        })

        start_idx += step

    return pd.DataFrame(metrics)


if __name__ == "__main__":
    prices = download_price_data(
        tickers=("AAPL", "MSFT", "GOOGL"),
        start="2015-01-01",
        end="2025-12-02",
        interval="1d",
    )

    df_metrics = rolling_evaluation(prices)
    print(df_metrics)

    print("\nAverage metrics across windows:")
    print(df_metrics[[
        "joh_total_return", "bo_total_return",
        "joh_sharpe", "bo_sharpe",
        "joh_max_dd", "bo_max_dd",
    ]].mean())
