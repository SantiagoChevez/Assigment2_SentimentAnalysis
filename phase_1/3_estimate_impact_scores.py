import pandas as pd
import numpy as np
import os

def zscore_safe(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        # no variance to all zeros
        return pd.Series(0.0, index=s.index)   
    return (s - mu) / sd

def piecewise_impact_vec(zr: pd.Series, zsigma: pd.Series) -> pd.Series:
    zr = pd.to_numeric(zr, errors="coerce")
    zsigma = pd.to_numeric(zsigma, errors="coerce")

    # default to 0 when zr is NaN (insufficient data)
    score = np.zeros(len(zr), dtype=float)

    # only rows where we have a return z-score
    has_zr = zr.notna()

    # base magnitude: 1 for |zr| > 0.5
    base = (np.abs(zr) > 0.5).astype(int)

    # +1 if |zr| > 1 (large move)
    add_ret = (np.abs(zr) > 1).astype(int)

    # +1 if zsigma > 1 (high vol)
    add_vol = (zsigma > 1).astype(int)

    signed = np.sign(zr)

    score = np.where(
        has_zr & (base == 1),
        signed * (1 + add_ret + add_vol),
        0.0
    )

    # final as int, but protect NaNs (there shouldn’t be any after where)
    return pd.Series(score).fillna(0).astype(int)

def estimate_impact_scores():
    # input_path = "../datasets/historical_prices_with_volatility.csv"
    input_path = "datasets/historical_prices_with_volatility.csv"
    # output_path = "../datasets/historical_prices_impact.csv"
    output_path = "datasets/historical_prices_impact.csv"

    df = pd.read_csv(input_path)
    df.sort_values(["symbol", "date"], inplace=True)

    # z-scores by symbol (robust to zero-variance)
    df["z_return"] = df.groupby("symbol", group_keys=False)["daily_return"].transform(zscore_safe)
    df["z_vol"]    = df.groupby("symbol", group_keys=False)["daily_volatility"].transform(zscore_safe)

    # vectorized piecewise impact (NaNs -> 0)
    df["impact_score"] = piecewise_impact_vec(df["z_return"], df["z_vol"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    
    print(df.head(10))
    print(f"\n Saved impact scores to {output_path}")

if __name__ == "__main__":
    estimate_impact_scores()
