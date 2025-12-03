from __future__ import annotations

import numpy as np
import pandas as pd
import optuna

from .data_loader import download_price_data
from .cointegration import get_baseline_weights, construct_spread
from .backtest import backtest_mean_reversion


def normalize_weights(raw_weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights so that sum(abs(weights)) = 1.
    This prevents BO from just scaling everything up/down to game the Sharpe.
    """
    s = np.sum(np.abs(raw_weights))
    if s == 0:
        # avoid division by zero; fall back to equal weights
        return np.ones_like(raw_weights) / len(raw_weights)
    return raw_weights / s


def optimize_weights_with_bo(
    prices: pd.DataFrame,
    n_trials: int = 50,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.0,
    lookback: int = 60,
) -> tuple[optuna.Study, pd.Series, dict]:
    """
    Run Bayesian Optimization to find basket weights that
    maximize Sharpe ratio of the mean-reversion strategy,
    keeping thresholds fixed.
    """
    tickers = list(prices.columns)
    n_assets = len(tickers)

    def objective(trial: optuna.Trial) -> float:
        # Sample raw weights in some reasonable range
        raw_weights = np.array([
            trial.suggest_float(f"w_{i}", -2.0, 2.0) for i in range(n_assets)
        ])

        # Normalize to avoid scale issues
        weights_vec = normalize_weights(raw_weights)
        weights = pd.Series(weights_vec, index=tickers)

        # Construct spread and run backtest
        spread = construct_spread(prices, weights)
        results = backtest_mean_reversion(
            spread,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            lookback=lookback,
            initial_capital=100_000.0,
        )

        sharpe = results["stats"]["sharpe"]

        # Sometimes Sharpe can be nan or inf if no trades happen etc.
        if not np.isfinite(sharpe):
            return -1e6  # big penalty

        return sharpe

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Build best weights as a Series
    best_params = study.best_params
    best_raw = np.array([best_params[f"w_{i}"] for i in range(n_assets)])
    best_vec = normalize_weights(best_raw)
    best_weights = pd.Series(best_vec, index=tickers, name="bo_weights")

    # Evaluate best weights with a full backtest
    best_spread = construct_spread(prices, best_weights)
    best_results = backtest_mean_reversion(
        best_spread,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        lookback=lookback,
        initial_capital=100_000.0,
    )

    return study, best_weights, best_results


def optimize_strategy_with_bo(
    prices: pd.DataFrame,
    n_trials: int = 80,
    lookback: int = 60,
) -> tuple[optuna.Study, pd.Series, dict]:
    """
    Bayesian Optimization over BOTH:
      - basket weights
      - trading thresholds (entry_z, exit_z, stop_z)

    Objective: maximize Sharpe ratio on the backtest.
    """
    tickers = list(prices.columns)
    n_assets = len(tickers)

    def objective(trial: optuna.Trial) -> float:
        # 1) Sample raw weights
        raw_weights = np.array([
            trial.suggest_float(f"w_{i}", -2.0, 2.0) for i in range(n_assets)
        ])
        weights_vec = normalize_weights(raw_weights)
        weights = pd.Series(weights_vec, index=tickers)

        # 2) Sample trading thresholds
        entry_z = trial.suggest_float("entry_z", 1.0, 3.0)
        exit_z = trial.suggest_float("exit_z", 0.1, 1.5)
        # stop_z must be >= entry_z (otherwise nonsense)
        stop_z = trial.suggest_float("stop_z", entry_z, 4.0)

        spread = construct_spread(prices, weights)

        results = backtest_mean_reversion(
            spread,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            lookback=lookback,
            initial_capital=100_000.0,
        )

        sharpe = results["stats"]["sharpe"]
        if not np.isfinite(sharpe):
            return -1e6

        return sharpe

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Extract best weights + thresholds
    best_params = study.best_params
    best_raw = np.array([best_params[f"w_{i}"] for i in range(n_assets)])
    best_vec = normalize_weights(best_raw)
    best_weights = pd.Series(best_vec, index=tickers, name="bo_strategy_weights")

    best_entry_z = best_params["entry_z"]
    best_exit_z = best_params["exit_z"]
    best_stop_z = best_params["stop_z"]

    best_spread = construct_spread(prices, best_weights)
    best_results = backtest_mean_reversion(
        best_spread,
        entry_z=best_entry_z,
        exit_z=best_exit_z,
        stop_z=best_stop_z,
        lookback=lookback,
        initial_capital=100_000.0,
    )

    best_results["best_thresholds"] = {
        "entry_z": best_entry_z,
        "exit_z": best_exit_z,
        "stop_z": best_stop_z,
    }

    return study, best_weights, best_results


if __name__ == "__main__":
    # 1) Load data
    prices = download_price_data(
        tickers=("AAPL", "MSFT", "GOOGL"),
        start="2020-01-01",
        end="2025-12-02",
        interval="1d",
    )

    # 2) Baseline Johansen
    joh_weights = get_baseline_weights(prices)
    joh_spread = construct_spread(prices, joh_weights)
    joh_results = backtest_mean_reversion(
        joh_spread,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        lookback=60,
        initial_capital=100_000.0,
    )

    print("[bayes_opt] Johansen baseline weights:")
    print(joh_weights)
    print("\n[bayes_opt] Johansen performance:")
    for k, v in joh_results["stats"].items():
        print(f"{k:>15}: {v:.4f}")

    # 3) Bayesian Optimization (weights only)
    print("\n[bayes_opt] Running Bayesian Optimization (weights only)...")
    study, bo_weights, bo_results = optimize_weights_with_bo(
        prices,
        n_trials=50,   # can lower for speed
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        lookback=60,
    )

    print("\n[bayes_opt] Best Sharpe from BO study:", study.best_value)
    print("[bayes_opt] BO-optimized weights:")
    print(bo_weights)

    print("\n[bayes_opt] BO-optimized performance:")
    for k, v in bo_results["stats"].items():
        print(f"{k:>15}: {v:.4f}")

    # 4) Full strategy optimization: weights + thresholds
    print("\n[bayes_opt] Running full strategy BO (weights + thresholds)...")
    strat_study, strat_weights, strat_results = optimize_strategy_with_bo(
        prices,
        n_trials=80,
        lookback=60,
    )

    print("\n[bayes_opt] Best Sharpe from strategy BO study:", strat_study.best_value)
    print("[bayes_opt] Strategy-BO optimized weights:")
    print(strat_weights)

    print("\n[bayes_opt] Strategy-BO best thresholds:")
    print(strat_results["best_thresholds"])

    print("\n[bayes_opt] Strategy-BO optimized performance:")
    for k, v in strat_results["stats"].items():
        print(f"{k:>15}: {v:.4f}")
