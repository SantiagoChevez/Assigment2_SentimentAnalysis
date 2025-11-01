"""
Trading simulation + evaluation harness (phase 3.2)

Loads the phase_3/1_trading_rules.py module using importlib (since the
filename starts with a digit and can't be imported with standard syntax),
then uses its functions (e.g., model_predict, trading_rules) to run a
simple backtest over prepared data.
"""

from pathlib import Path
import importlib.util
import sys

# --- Import 1_trading_rules.py via importlib ---------------------------------
THIS_DIR = Path(__file__).resolve().parent
RULES_PATH = THIS_DIR / "1_trading_rules.py"

if not RULES_PATH.exists():
	raise FileNotFoundError(f"Trading rules file not found: {RULES_PATH}")

_spec = importlib.util.spec_from_file_location("trading_rules", str(RULES_PATH))
trading_rules = importlib.util.module_from_spec(_spec)
sys.modules["trading_rules"] = trading_rules
assert _spec.loader is not None
_spec.loader.exec_module(trading_rules)

# Now you can call:
#   trading_rules.model_predict(df)
#   trading_rules.trading_rules(row)

# logic using function/class calls from 1_trading_rules.py
# 3.2 handles loading the dataset as df and passing it to 1_trading_rules.model_predict() for evaluation
# 1_trading_rules.model_predict(df) --> outputs pred_impact_score (need to check format returned from model inference)
# add pred_impact_score from model inference to df : df['pred_impact_score'] = model_predict(df) 
# map stock name, price and date to vectorized dataset
# map after model inference: easier to combine datasets into one df for simulation
# iterate through each row in df and call 1_trading_rules.trading_rules(<df row which is probably formatted as a Series>)to determine buy/sell/hold

# make sure to initialize list to keep track of balance
# trade tracking and metric calculations can go in a new df 
# can store data from each trade in df and ouput final to csv or log to file after the end of each trade