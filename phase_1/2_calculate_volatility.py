import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

MARKET_SYMBOL = "s&p"

def calc_daily_log_returns(df):
    """
    Calculate daily logarithmic returns for all symbols.

    Computes r_t = ln(P_t / P_{t-1}) for each symbol, where P_t is the closing price
    at time t. Results are stored in the 'daily_return' column (in-place modification).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing historical prices with 'symbol' and 'close' columns.
        Must be sorted by symbol and date. Modified in-place.

    Returns:
    --------
    None
        Modifies the DataFrame in-place by adding 'daily_return' column.
    """
    df["daily_return"] = df.groupby("symbol")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )


def calc_daily_volatility(df):
    """
    Calculate rolling daily volatility (3-day window) from daily log returns.

    Computes the rolling standard deviation of daily returns using a 3-day window.
    Volatility is defined as the standard deviation of logarithmic returns.
    Results are stored in the 'daily_volatility' column (in-place modification).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing historical prices with 'symbol' and 'daily_return' columns.
        Modified in-place.

    Returns:
    --------
    None
        Modifies the DataFrame in-place by adding 'daily_volatility' column.
    """

    df["daily_volatility"] = (
        df.groupby("symbol")["daily_return"]
        .transform(lambda s: s.rolling(3, min_periods=3).std(ddof=1))
    )

def market_return(df):
    """
    Map market returns (S&P 500) to all rows in the DataFrame by date.

    Extracts daily returns for the market symbol (typically 's&p') and maps them
    to all symbols' rows based on matching dates. Results are stored in the
    'market_return' column (in-place modification).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing historical prices with 'symbol', 'date', and 'daily_return' columns.
        Must contain rows for the market symbol (MARKET_SYMBOL). Modified in-place.

    Returns:
    --------
    None
        Modifies the DataFrame in-place by adding 'market_return' column.
    """
    df['date'] = pd.to_datetime(df['date'])
    market_log_returns = (
        df[df['symbol'] == MARKET_SYMBOL][['date', 'daily_return']]
        .drop_duplicates(subset=['date'])
        .set_index('date')['daily_return']
    )
    df['market_return'] = df['date'].map(market_log_returns)


def fit_one(g):
    """
    Fit market model (CAPM) for a single symbol group and compute alpha, beta.

    Fits a linear regression: daily_return = alpha + beta * market_return + epsilon.
    For the market symbol itself, alpha=0, beta=1 by definition.
    Also computes market-adjusted returns and idiosyncratic returns.

    Parameters:
    -----------
    g : pd.DataFrame
        A group from groupby operation, containing data for a single symbol.
        Must have 'daily_return' and 'market_return' columns.

    Returns:
    --------
    pd.DataFrame
        Same group with added columns: 'alpha', 'beta', 'market_adj_return', 'idiosyn_return'.
    """
    if g.name == MARKET_SYMBOL:
        g["alpha"] = 0.0
        g["beta"] = 1.0
        g["market_adj_return"] = 0.0
        g["idiosyn_return"] = 0.0
        return g

    fit = g.dropna(subset=["daily_return", "market_return"])
    if len(fit) < 2 or np.isclose(fit["market_return"].var(ddof=1), 0.0):
        a, b = 0.0, 1.0
    else:
        res = smf.ols("daily_return ~ market_return", data=fit).fit()
        a = float(res.params.get("Intercept", 0.0))
        b = float(res.params.get("market_return", 1.0))

    g["alpha"] = a
    g["beta"] = b
    g["market_adj_return"] = g["daily_return"] - (a + b * g["market_return"])
    g["idiosyn_return"] = g["market_adj_return"]
    return g


def calc_market_adj_return(df):
    """
    Calculate market-adjusted returns using the market model (CAPM).

    Fits alpha and beta for each symbol using OLS regression, then computes:
    - market_adj_return = daily_return - (alpha + beta * market_return)
    - idiosyn_return = market_adj_return (by definition)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'symbol', 'daily_return', and 'market_return' columns.
        Modified in-place.

    Returns:
    --------
    None
        Modifies the DataFrame in-place by adding 'alpha', 'beta', 'market_adj_return',
        and 'idiosyn_return' columns.
    """
    result = df.groupby("symbol", group_keys=False).apply(fit_one)
    for col in ["alpha", "beta", "market_adj_return", "idiosyn_return"]:
        df[col] = result[col]


def calc_market_adj_volatility(df):
    """
    Calculate rolling volatility of market-adjusted returns (3-day window).

    Computes the rolling standard deviation of market-adjusted returns using a 3-day window.
    Market-adjusted return volatility measures the volatility of returns after accounting
    for market movements.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'symbol' and 'market_adj_return' columns.
        Modified in-place.

    Returns:
    --------
    None
        Modifies the DataFrame in-place by adding 'market_adj_volatility' column.
    """
    df["market_adj_volatility"] = (
        df.groupby("symbol")["market_adj_return"]
        .transform(lambda s: s.rolling(3, min_periods=3).std(ddof=1))
    )    


def calc_idiosyn_volatility(df):
    """
    Calculate rolling idiosyncratic volatility (3-day window).

    Computes the rolling standard deviation of idiosyncratic returns using a 3-day window.
    Idiosyncratic volatility measures asset-specific risk independent of market movements.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'symbol' and 'idiosyn_return' columns.
        Modified in-place.

    Returns:
    --------
    None
        Modifies the DataFrame in-place by adding 'idiosyn_volatility' column.
    """
    df["idiosyn_volatility"] = (
        df.groupby("symbol")["idiosyn_return"]
        .transform(lambda s: s.rolling(3, min_periods=3).std(ddof=1))
    )

def calc_asset_volatility(df):
    """
    Calculate total asset volatility (overall standard deviation of returns).

    Computes the standard deviation of daily returns for each symbol across the entire
    time period. This is a constant value per symbol (not rolling).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'symbol' and 'daily_return' columns.
        Modified in-place.

    Returns:
    --------
    None
        Modifies the DataFrame in-place by adding 'asset_volatility' column.
    """
    df["asset_volatility"] = (
        df.groupby("symbol")["daily_return"]
        .transform(lambda s: s.std(ddof=1))
    )

def calculate_volatility():
    """
    Main function to calculate all volatility metrics for historical prices.

    Reads historical prices from CSV, computes all volatility measures including:
    - Daily log returns and volatility
    - Market returns and market-adjusted returns
    - Alpha and beta coefficients (CAPM)
    - Idiosyncratic returns and volatility
    - Asset-level volatility

    Parameters:
    -----------
    None
        Reads from datasets/historical_prices.csv relative to project root.

    Returns:
    --------
    pd.DataFrame
        DataFrame with all calculated volatility metrics. Columns include:
        date, symbol, open, high, low, close, volume, daily_return, daily_volatility,
        market_return, beta, alpha, idiosyn_return, idiosyn_volatility,
        market_adj_return, market_adj_volatility, asset_volatility

    Raises:
    -------
    FileNotFoundError
        If historical_prices.csv is not found in the datasets directory.
    """
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / 'datasets' / 'historical_prices.csv'
    if not input_path.exists():
        raise FileNotFoundError(
            f"Historical prices file not found at: {input_path}\n"
            "Ensure the 'datasets' folder exists in the project root."
        )

    df = pd.read_csv(str(input_path))
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['symbol', 'date'], inplace=True)

    calc_daily_log_returns(df)
    calc_daily_volatility(df)
    market_return(df)
    calc_market_adj_return(df)
    calc_market_adj_volatility(df)
    calc_idiosyn_volatility(df)
    calc_asset_volatility(df)

    print(df.head(10))

    required_cols = [
        'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'daily_return', 'daily_volatility', 'market_return', 'beta', 'alpha',
        'idiosyn_return', 'idiosyn_volatility', 'market_adj_return', 'market_adj_volatility'
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[required_cols]
    return df

if __name__ == "__main__":
    """Execute volatility calculations and save results to CSV."""
    df_result = calculate_volatility()
    output_path = Path(__file__).resolve().parent.parent / 'datasets' / 'historical_prices_with_volatility.csv'
    df_result.to_csv(str(output_path), index=False)
    print(f"\n[SUCCESS] Saved results to: {output_path}")
