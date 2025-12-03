from src.data_loader import download_price_data
from src.cointegration import get_baseline_weights, construct_spread
from src.backtest import backtest_mean_reversion, compute_zscore
from src.plotting import plot_equity_curves, plot_spread_and_zscore
from src.bayes_opt import optimize_weights_with_bo, optimize_strategy_with_bo

# 1. Load prices
prices = download_price_data(("AAPL", "MSFT", "GOOGL"), "2020-01-01", "2025-12-02")

# ---------------------------------------------------------------
# 2. JOHANSEN BASELINE
# ---------------------------------------------------------------
joh_weights = get_baseline_weights(prices)
joh_spread = construct_spread(prices, joh_weights)
joh_results = backtest_mean_reversion(joh_spread)
joh_equity = joh_results["equity_curve"]

# ---------------------------------------------------------------
# 3. BAYESIAN OPTIMIZATION (WEIGHTS ONLY)
# ---------------------------------------------------------------
bo_study, bo_weights, bo_results = optimize_weights_with_bo(
    prices,
    n_trials=40,   # use low number for fast test
    entry_z=2.0,
    exit_z=0.5,
    stop_z=3.0,
    lookback=60,
)
bo_equity = bo_results["equity_curve"]

# ---------------------------------------------------------------
# 4. FULL STRATEGY BO (weights + thresholds)
# ---------------------------------------------------------------
st_study, st_weights, st_results = optimize_strategy_with_bo(
    prices,
    n_trials=40,   # also low for quick test
    lookback=60,
)
st_equity = st_results["equity_curve"]

# ---------------------------------------------------------------
# 5. PLOT EQUITY CURVES (ALL THREE)
# ---------------------------------------------------------------
plot_equity_curves({
    "Johansen": joh_equity,
    "BO Weights": bo_equity,
    "Strategy BO": st_equity,
})

# ---------------------------------------------------------------
# 6. OPTIONAL: SHOW SPREAD/Z FOR STRATEGY BO
# ---------------------------------------------------------------
spread_bo = construct_spread(prices, st_weights)
z = compute_zscore(spread_bo)
plot_spread_and_zscore(
    spread_bo,
    z,
    entry_z=st_results["best_thresholds"]["entry_z"],
    exit_z=st_results["best_thresholds"]["exit_z"]
)
