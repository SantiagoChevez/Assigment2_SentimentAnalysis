#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets"
IN_PRICES = DATA_DIR / "historical_prices.csv"
OUT_IMPACT = DATA_DIR / "historical_prices_impact.csv"

# Configuration constants
MARKET_SYMBOL = "s&p"   # Market proxy symbol (maps from ^GSPC or SPY)
ROLL_N = 3               # Rolling window size for volatility calculations (days)


def log_ret(x: pd.Series) -> pd.Series:
    """
    Calculate logarithmic returns for a price series.

    Computes r_t = ln(P_t / P_{t-1}) where P_t is the price at time t.

    Parameters:
    -----------
    x : pd.Series
        Price series (typically closing prices).

    Returns:
    --------
    pd.Series
        Logarithmic returns. First value will be NaN due to lag.
    """
    return np.log(x / x.shift(1))


def fit_alpha_beta(r_asset: pd.Series, r_mkt: pd.Series) -> tuple[float, float]:
    """
    Fit market model (CAPM) to estimate alpha and beta coefficients.

    Uses OLS regression: r_asset = alpha + beta * r_market + epsilon.
    Returns default values (alpha=0, beta=1) if insufficient data or zero variance.

    Parameters:
    -----------
    r_asset : pd.Series
        Asset daily returns series.
    r_mkt : pd.Series
        Market daily returns series (must align with r_asset).

    Returns:
    --------
    tuple[float, float]
        (alpha, beta) coefficients. Defaults to (0.0, 1.0) if regression cannot be performed.
    """
    idx = r_asset.dropna().index.intersection(r_mkt.dropna().index)
    if len(idx) < 2:
        return 0.0, 1.0

    a = r_asset.loc[idx]
    m = r_mkt.loc[idx]
    var_m = np.var(m, ddof=1)
    if not np.isfinite(var_m) or var_m == 0:
        return 0.0, 1.0

    cov_am = np.cov(a, m, ddof=1)[0, 1]
    beta = cov_am / var_m
    alpha = a.mean() - beta * m.mean()
    return alpha, beta


def zscore_per_group(s: pd.Series) -> pd.Series:
    """
    Calculate z-scores (standardized values) for a series.

    Computes (x - mean) / std for each value. Returns zeros if std is zero or infinite.

    Parameters:
    -----------
    s : pd.Series
        Series to standardize.

    Returns:
    --------
    pd.Series
        Z-scores with same index as input. Returns zeros if std is invalid.
    """
    mu = s.mean()
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def score_from_z(z_r: float, z_sig: float) -> int:
    """
    Map z-scores to discrete impact score in range [-3, 3].

    Scoring logic:
    - Neutral (|z_r| <= 0.5): returns 0
    - Base score starts at 1 (magnitude)
    - Increases by 1 if |z_r| > 1
    - Increases by 1 if volatility z-score > 1
    - Sign preserved, clamped to [-3, 3]

    Parameters:
    -----------
    z_r : float
        Z-score of idiosyncratic return.
    z_sig : float
        Z-score of daily volatility.

    Returns:
    --------
    int
        Impact score in range [-3, -2, -1, 0, 1, 2, 3]. Returns 0 for invalid inputs.
    """
    if not np.isfinite(z_r) or not np.isfinite(z_sig):
        return 0
    if abs(z_r) <= 0.5:
        return 0

    base = 1
    if abs(z_r) > 1:
        base += 1
    if z_sig > 1:
        base += 1

    return int(np.sign(z_r) * max(1, min(3, base)))

def main():
    """
    Main function to estimate impact scores from historical prices.

    Processes the full pipeline: calculates returns, fits market model, computes
    volatilities and z-scores, and maps to discrete impact scores.

    Raises:
    -------
    FileNotFoundError
        If historical_prices.csv is not found.
    ValueError
        If market symbol (MARKET_SYMBOL) is not found in the data.
    """
    if not IN_PRICES.exists():
        raise FileNotFoundError(f"Missing input: {IN_PRICES}")

    # Determine which price columns are available
    sample_df = pd.read_csv(IN_PRICES, nrows=0)
    usecols = ["date", "symbol", "open", "high", "low", "volume"]
    price_cols_to_read = []
    if "AdjClose" in sample_df.columns:
        price_cols_to_read.append("AdjClose")
    if "close" in sample_df.columns:
        price_cols_to_read.append("close")
    usecols.extend(price_cols_to_read)

    # Read data with optimized dtypes for memory efficiency
    df = pd.read_csv(
        IN_PRICES,
        usecols=usecols,
        parse_dates=["date"],
        dtype={
            "symbol": "category",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "volume": "float64",
        },
    ).sort_values(["symbol", "date"]).reset_index(drop=True)

    # Normalize price column: merge 'close' and 'AdjClose' into 'AdjClose'
    if 'close' in df.columns:
        if 'AdjClose' not in df.columns:
            df.rename(columns={'close': 'AdjClose'}, inplace=True)
        else:
            df['AdjClose'] = pd.to_numeric(df['AdjClose'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['AdjClose'] = df['AdjClose'].fillna(df['close']).astype('float32')
    else:
        if 'AdjClose' in df.columns:
            df['AdjClose'] = pd.to_numeric(df['AdjClose'], errors='coerce').astype('float32')

    # Step 1: Calculate daily log returns per symbol
    if 'close' in df.columns and df['close'].notna().any():
        price_col = 'close'
    else:
        price_col = 'AdjClose'
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df["daily_return"] = df.groupby("symbol", observed=True)[price_col].transform(log_ret).astype("float32")

    # Step 2: Extract and merge market returns (S&P 500)
    mkt = df[df["symbol"].astype(str).str.lower() == MARKET_SYMBOL.lower()][["date", "daily_return"]]
    if mkt.empty:
        raise ValueError(f"Market series '{MARKET_SYMBOL}' not found in {IN_PRICES.name}.")
    mkt = mkt.rename(columns={"daily_return": "market_return"})
    df = df.merge(mkt, on="date", how="left")
    df["market_return"] = df["market_return"].astype("float32")

    # Step 3: Fit market model (alpha, beta) per symbol
    ab = {}
    for sym, g in df.groupby("symbol", observed=True):
        s = str(sym)
        if s.lower() == MARKET_SYMBOL.lower():
            ab[sym] = (0.0, 1.0)
            continue
        r_a = g.set_index("date")["daily_return"]
        r_m = g.set_index("date")["market_return"]
        alpha, beta = fit_alpha_beta(r_a, r_m)
        ab[sym] = (alpha, beta)

    df["alpha"] = df["symbol"].map(lambda s: ab.get(s, (np.nan, np.nan))[0]).astype("float32")
    df["beta"] = df["symbol"].map(lambda s: ab.get(s, (np.nan, np.nan))[1]).astype("float32")

    # Step 4: Calculate idiosyncratic returns and rolling volatilities
    df["idiosyn_return"] = (df["daily_return"] - (df["alpha"] + df["beta"] * df["market_return"])).astype("float32")

    roll_std = lambda s: s.rolling(ROLL_N, min_periods=3).std(ddof=1).astype("float32")
    df["daily_volatility"] = df.groupby("symbol", observed=True)["daily_return"].transform(roll_std)
    df["idiosyn_volatility"] = df.groupby("symbol", observed=True)["idiosyn_return"].transform(roll_std)

    # Create aliases for downstream compatibility
    df["market_adj_return"] = df["idiosyn_return"]
    df["market_adj_volatility"] = df["idiosyn_volatility"]

    # Step 5: Calculate z-scores per symbol
    df["z_r"] = df.groupby("symbol", observed=True)["idiosyn_return"].transform(zscore_per_group).astype("float32")
    df["z_sigma"] = df.groupby("symbol", observed=True)["daily_volatility"].transform(zscore_per_group).astype("float32")

    # Step 6: Map z-scores to discrete impact scores
    df["impact_score"] = [score_from_z(r, s) for r, s in zip(df["z_r"], df["z_sigma"])]

    # Step 7: Normalize output columns
    if 'close' in df.columns and df['close'].notna().any():
        pass  # Keep existing 'close' column
    else:
        df['close'] = df['AdjClose']

    # Select and order output columns
    out_cols = [
        "date", "symbol", "open", "high", "low", "close", "volume",
        "daily_return", "daily_volatility",
        "market_return", "beta", "alpha",
        "idiosyn_return", "idiosyn_volatility",
        "market_adj_return", "market_adj_volatility",
        "impact_score",
    ]
    out = df[out_cols].copy()

    # Normalize for compatibility with downstream processing
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Write output
    OUT_IMPACT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_IMPACT, index=False)
    print(f"Wrote {OUT_IMPACT} | rows={len(out)} | symbols={out['symbol'].nunique()}")


if __name__ == "__main__":
    main()