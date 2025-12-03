
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def run_johansen(
    price_df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """
    Run the Johansen cointegration test on a DataFrame of prices.

    Args:
        price_df: DataFrame (T x N) of price series (columns = tickers).
        det_order: Deterministic trend term (0 = no trend in data).
        k_ar_diff: Number of lag differences in the VECM.

    Returns:
        dict with eigenvalues, eigenvectors, and test statistics.
    """
    # Make sure no NaNs (Johansen will complain)
    if price_df.isna().any().any():
        price_df = price_df.dropna()

    # Johansen expects a numpy array of log prices
    log_prices = np.log(price_df.values)

    result = coint_johansen(log_prices, det_order, k_ar_diff)

    return {
        "eigenvalues": result.eig,        # eigenvalues for cointegration strength
        "eigenvectors": result.evec,      # columns = cointegrating vectors
        "lr1": result.lr1,                # trace test statistics
        "lr2": result.lr2,                # max eigenvalue test statistics
        "crit_trace": result.cvt,         # critical values (trace)
        "crit_max_eig": result.cvm,       # critical values (max eigen)
    }


def get_baseline_weights(price_df: pd.DataFrame) -> pd.Series:
    """
    Compute baseline cointegrating weights using the Johansen test.

    Strategy:
        - Run Johansen on log prices.
        - Take the first eigenvector as the cointegrating vector.
        - Normalize so that the first asset has weight 1.

    Returns:
        Pandas Series with index = tickers, values = weights.
    """
    res = run_johansen(price_df)
    eigenvectors = res["eigenvectors"]  # shape (N_assets, N_assets)
    tickers = list(price_df.columns)

    # Take first cointegrating vector (first column)
    beta = eigenvectors[:, 0]

    # Normalize so that the first asset has weight exactly 1
    if beta[0] != 0:
        beta = beta / beta[0]

    weights = pd.Series(beta, index=tickers, name="johansen_weights")
    return weights


def construct_spread(price_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Construct the spread time series:
        spread_t = sum_i (w_i * log(price_i_t))

    Args:
        price_df: DataFrame with price columns matching weights.index.
        weights: Series of weights indexed by ticker symbol.

    Returns:
        Pandas Series of spread values over time.
    """
    # Align columns
    aligned = price_df[weights.index].dropna()
    log_prices = np.log(aligned)

    spread = log_prices.mul(weights, axis=1).sum(axis=1)
    spread.name = "spread"
    return spread


if __name__ == "__main__":
    # Minimal demo if you run this file directly
    from .data_loader import download_price_data

    df_prices = download_price_data(
        tickers=("AAPL", "MSFT", "GOOGL"),
        start="2020-01-01",
        end="2025-12-31",
        interval="1d",
    )

    print("[cointegration] Price data head:")
    print(df_prices.head())

    weights = get_baseline_weights(df_prices)
    print("\n[cointegration] Johansen baseline weights:")
    print(weights)

    spread = construct_spread(df_prices, weights)
    print("\n[cointegration] Spread (last 5 values):")
    print(spread.tail())
