from pathlib import Path
import importlib.util
import sys
import pandas as pd
import torch

# Import 1_trading_rules.py via importlib
THIS_DIR = Path(__file__).resolve().parent
RULES_PATH = THIS_DIR / "1_trading_rules.py"

if not RULES_PATH.exists():
	raise FileNotFoundError(f"Trading rules file not found: {RULES_PATH}")

_spec = importlib.util.spec_from_file_location("trading_rules", str(RULES_PATH))
trading_rules = importlib.util.module_from_spec(_spec)
sys.modules["trading_rules"] = trading_rules
assert _spec.loader is not None
_spec.loader.exec_module(trading_rules)

REPO_ROOT = THIS_DIR.parent
DATASETS_DIR = REPO_ROOT / "datasets"

def get_prices(df):
	prices = df.copy()

def process_datasets():
    df_prices = pd.read_csv(DATASETS_DIR / "historical_prices.csv")
    return df_prices
    

# logic using function/class calls from 1_trading_rules.py
# map stock name, price and date to vectorized dataset
# map after model inference: easier to combine datasets into one df for simulation
# iterate through each row in df and call 1_trading_rules.trading_rules(<df row which is probably formatted as a Series>)to determine buy/sell/hold
def run_simulation():
	# process datasets
	df_prices = process_datasets()

    # get model predictions
	df_with_preds = trading_rules.model_predict(model_key="tfidf")

    # merge prices with predictions data on symbol and date
	stock_data = df_with_preds.merge(
        df_prices[['symbol', 'date', 'AdjClose']],
        on=['symbol', 'date'],
        how='left'
    )
	print(stock_data[stock_data['AdjClose'].isna()])


# make sure to initialize list to keep track of balance
# trade tracking and metric calculations can go in a new df 
# can store data from each trade in df and ouput final to csv or log to file after the end of each trade

# append state columns for state tracking
if __name__ == "__main__":
	run_simulation()
	
