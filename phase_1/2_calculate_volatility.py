import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

MARKET_SYMBOL = "s&p"

def calc_daily_log_returns(df):
    """
    Calculate daily logarithmic returns from a df with Adjusted Closing prices.

    Parameters:
    df: dataframe containing historical prices 

    Returns: none

    """
    # First calculate for all symbols using AdjClose
    df["daily_return"] = df.groupby("symbol")["AdjClose"].transform(
        lambda s: np.log(s / s.shift(3))
    )
    
    # Then overwrite S&P values using 'close' column (since S&P's AdjClose is NaN)
    sp_mask = df["symbol"] == MARKET_SYMBOL
    df.loc[sp_mask, "daily_return"] = np.log(
        df.loc[sp_mask, "close"] / df.loc[sp_mask, "close"].shift(3)
    )


def calc_daily_volatility(df):
    """
    Calculate daily volatility from a df with daily log returns.

    Volatility is defined as the standard deviation of the logarithmic returns.

    Parameters:
    df: dataframe containing historical prices with daily_log_return column
    

    Returns: none
    """

    df["daily_volatility"] = (
        df.groupby("symbol")["daily_return"]
        .transform(lambda s: s.rolling(3, min_periods=3).std(ddof=1))
    )

def market_return(df):
    """
    Calculate market return volatility.

    Market return volatility is defined as the standard deviation of the market's logarithmic returns.

    Parameters:
 

    Returns: none

    """
    market_log_returns = df[df['symbol'] == MARKET_SYMBOL][['date', 'daily_return']]
    df['market_return'] = df['date'].map(market_log_returns.set_index('date')['daily_return'])


def fit_one(g):
    """
    Fit a linear regression model for each symbol in the dataframe.

    Parameters:
    df: dataframe containing historical prices with daily_return and market_return columns

    Returns: none

    """
    if g.name == 's&p':
        g["alpha"] = 0.0
        g["beta"]  = 1.0
        g["idiosyn_return"] = 0.0
        return g
    fit = g.dropna(subset=["daily_return","market_return"])
    # Check if we have enough data after dropping NaNs
    if len(fit) < 2 or np.isclose(fit["market_return"].var(ddof=1), 0.0):
        # Not enough data or no variance in market returns - use defaults
        a, b = 0.0, 1.0
    else:
        res = smf.ols("daily_return ~ market_return", data=fit).fit()
        a = float(res.params["Intercept"])
        b = float(res.params["market_return"])
    g["alpha"] = a
    g["beta"]  = b
    g["market_adj_return"] = g["daily_return"] - (g["alpha"] + g["beta"]*g["market_return"])
    return g


def calc_market_adj_return(df):
    """
    Calculate market-adjusted return volatility.

    Market-adjusted return volatility is defined as the standard deviation of the difference
    between the logarithmic returns of the asset and the market.

    Parameters:
 

    Returns: none

    """
    # Apply fit_one to each group and update df in-place
    result = df.groupby("symbol", group_keys=False).apply(fit_one)
    df[["alpha", "beta", "market_adj_return"]] = result[["alpha", "beta", "market_adj_return"]]

    

    
def calc_market_adj_volatility(df):
    """
    Calculate market-adjusted return volatility.

    Market-adjusted return volatility is defined as the standard deviation of the difference
    between the logarithmic returns of the asset and the market.

    Parameters:
 

    Returns: none

    """
    df["market_adj_volatility"] = (
        df.groupby("symbol")["market_adj_return"]
        .transform(lambda s: s.rolling(3, min_periods=3).std(ddof=1))
    )    

def calc_asset_volatility(df):
    """
    Calculate asset volatility.

    Asset volatility is defined as the overall standard deviation of the logarithmic returns 
    for each asset across the entire time period.

    Parameters:
 

    Returns: none

    """
    df["asset_volatility"] = (
        df.groupby("symbol")["daily_return"]
        .transform(lambda s: s.std(ddof=1))
    )

def calculate_volatility():
    """
    Calculate the volatility of a list of prices.

    Volatility is defined as the standard deviation of the logarithmic returns.

    Parameters:


    Returns:
    float: The calculated volatility in csv format.
    """
    # df = pd.read_csv('../datasets/historical_prices.csv')
    df = pd.read_csv('datasets/historical_prices.csv')


    #Ensure the dataframe is sorted by symbol and date
    df.sort_values(['symbol', 'date'], inplace=True)
    calc_daily_log_returns(df)
    calc_daily_volatility(df)
    market_return(df)
    calc_market_adj_return(df)
    calc_market_adj_volatility(df)
    calc_asset_volatility(df)
    print(df.head(10))

    return df

if __name__ == "__main__":
    df_result = calculate_volatility()
    
    # Save to CSV file
    # output_path = '../datasets/historical_prices_with_volatility.csv'
    output_path = 'datasets/historical_prices_with_volatility.csv'

    df_result.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to: {output_path}")
