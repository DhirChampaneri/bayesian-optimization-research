# src/data_loader.py

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_price_data(
    tickers: tuple[str, ...] | list[str] | str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted close prices for a set of tickers.

    Args:
        tickers: Single ticker (str) or list/tuple of tickers.
        start: Start date, e.g. "2015-01-01".
        end: End date, e.g. "2024-12-31".
        interval: Bar frequency, default "1d" (daily).

    Returns:
        DataFrame with Date index and one column per ticker containing prices.
    """
    # Normalize tickers to a list
    if isinstance(tickers, str):
        tickers = [tickers]

    # Download OHLCV data
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,  # adjust for splits/dividends
        progress=False,
    )

    # If multiple tickers, yfinance returns a multi-index: (field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        # Single ticker case: make it a DataFrame with one column
        close = data["Close"].to_frame()

    # Make sure column names are strings (tickers)
    close.columns = [str(c) for c in close.columns]

    # Drop rows where all prices are NaN
    close = close.dropna(how="all")

    return close


if __name__ == "__main__":
    # Quick manual test
    df = download_price_data(
        tickers=("AAPL", "MSFT", "GOOGL"),
        start="2018-01-01",
        end="2025-12-31",
    )
    print(df.tail())
