#!/usr/bin/env python3
"""
Phase 1 — Step 3: Estimate Impact Scores (single-pass pandas)

Input:  datasets/historical_prices.csv
Output: datasets/historical_prices_impact.csv

What it does:
1) Daily log returns per symbol
2) Merge market return (S&P) by date
3) Fit a simple market model per symbol (alpha, beta)
4) Compute idiosyncratic return and 3-day rolling volatilities
5) Per-symbol z-scores and a discrete impact_score in {-3..3}
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets"
IN_PRICES = DATA_DIR / "historical_prices.csv"
OUT_IMPACT = DATA_DIR / "historical_prices_impact.csv"

# ---- Config -----------------------------------------------------------------
MARKET_SYMBOL = "s&p"   # how ^GSPC is labeled in your collector
ROLL_N = 3              # “about three days” per assignment

# ---- Helpers ----------------------------------------------------------------
def log_ret(x: pd.Series) -> pd.Series:
    return np.log(x / x.shift(1))

def fit_alpha_beta(r_asset: pd.Series, r_mkt: pd.Series):
    # OLS on aligned indices: r_a = alpha + beta * r_m + e
    idx = r_asset.dropna().index.intersection(r_mkt.dropna().index)
    if len(idx) < 5:
        return np.nan, np.nan
    a = r_asset.loc[idx]
    m = r_mkt.loc[idx]
    var_m = np.var(m, ddof=1)
    if not np.isfinite(var_m) or var_m == 0:
        return np.nan, np.nan
    cov_am = np.cov(a, m, ddof=1)[0, 1]
    beta = cov_am / var_m
    alpha = a.mean() - beta * m.mean()
    return alpha, beta

def zscore_per_group(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def score_from_z(z_r: float, z_sig: float) -> int:
    # Neutral band, then step up magnitude with |z_r| and volatility
    if not np.isfinite(z_r) or not np.isfinite(z_sig):
        return 0
    if abs(z_r) <= 0.5:
        return 0
    base = 1
    if abs(z_r) > 1:
        base += 1
    if z_sig > 1:
        base += 1
    # clamp to [-3, 3]
    return int(np.sign(z_r) * max(1, min(3, base)))

# ---- Main -------------------------------------------------------------------
def main():
    if not IN_PRICES.exists():
        raise FileNotFoundError(f"Missing input: {IN_PRICES}")

    # Read once, with light dtypes to save memory
    usecols = ["date","symbol","open","high","low","close","volume"]
    df = pd.read_csv(
        IN_PRICES,
        usecols=usecols,
        parse_dates=["date"],
        dtype={
            "symbol": "category",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "float64",   # keep float for safety if large
        },
    ).sort_values(["symbol","date"]).reset_index(drop=True)

    # 1) Daily log returns per symbol
    df["daily_return"] = df.groupby("symbol", observed=True)["close"].transform(log_ret).astype("float32")

    # 2) Market series (S&P)
    mkt = df[df["symbol"].astype(str).str.lower() == MARKET_SYMBOL.lower()][["date","daily_return"]]
    if mkt.empty:
        raise ValueError(f"Market series '{MARKET_SYMBOL}' not found in {IN_PRICES.name}.")
    mkt = mkt.rename(columns={"daily_return": "market_return"})
    df = df.merge(mkt, on="date", how="left")
    df["market_return"] = df["market_return"].astype("float32")

    # 3) Alpha/Beta per symbol (constant over time)
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
    df["beta"]  = df["symbol"].map(lambda s: ab.get(s, (np.nan, np.nan))[1]).astype("float32")

    # 4) Idiosyncratic return and rolling volatilities
    df["idiosyn_return"] = (df["daily_return"] - (df["alpha"] + df["beta"] * df["market_return"])).astype("float32")

    roll_std = lambda s: s.rolling(ROLL_N, min_periods=2).std(ddof=1).astype("float32")
    df["daily_volatility"]   = df.groupby("symbol", observed=True)["daily_return"].transform(roll_std)
    df["idiosyn_volatility"] = df.groupby("symbol", observed=True)["idiosyn_return"].transform(roll_std)

    # Aliases (clarity for downstream)
    df["market_adj_return"]     = df["idiosyn_return"]
    df["market_adj_volatility"] = df["idiosyn_volatility"]

    # 5) Z-scores per symbol (on idiosyn_return and daily_volatility)
    df["z_r"]     = df.groupby("symbol", observed=True)["idiosyn_return"].transform(zscore_per_group).astype("float32")
    df["z_sigma"] = df.groupby("symbol", observed=True)["daily_volatility"].transform(zscore_per_group).astype("float32")

    # 6) Discrete impact score
    df["impact_score"] = [score_from_z(r, s) for r, s in zip(df["z_r"], df["z_sigma"])]

    # 7) Final selection & normalization for merges downstream
    out_cols = [
        "date","symbol","open","high","low","close","volume",
        "daily_return","daily_volatility",
        "market_return","beta","alpha",
        "idiosyn_return","idiosyn_volatility",
        "market_adj_return","market_adj_volatility",
        "impact_score",
    ]
    out = df[out_cols].copy()

    # Normalize to strings for join-compatibility with Phase 1.4
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["date"]   = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    OUT_IMPACT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_IMPACT, index=False)
    print(f"Wrote {OUT_IMPACT} | rows={len(out)} | symbols={out['symbol'].nunique()}")

if __name__ == "__main__":
    main()
