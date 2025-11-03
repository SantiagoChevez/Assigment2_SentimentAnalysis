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
	"""Normalize price column into `price` and ensure date parsing.

	This function will look for common price column names (case
	insensitive) and create a unified `price` column used by the
	simulator.
	"""
	df = df.copy()
	# look for common price column names (case-insensitive)
	price_candidates = ['AdjClose', 'adjclose', 'adj_close', 'adj_close', 'Close', 'close']
	found = None
	for c in price_candidates:
		if c in df.columns:
			found = c
			break
	# try a case-insensitive match if exact names weren't found
	if found is None:
		cols_lower = {col.lower(): col for col in df.columns}
		for cand in ['adjclose', 'adj_close', 'close']:
			if cand in cols_lower:
				found = cols_lower[cand]
				break
	if found is None:
		# fallback: try any numeric column that looks like a price (last resort)
		numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
		if len(numeric_cols) > 0:
			found = numeric_cols[-1]
		else:
			raise RuntimeError(f"No price column found in historical prices. Columns: {list(df.columns)}")

	# create normalized price column
	df['price'] = df[found]

	# ensure date column is datetime if present
	if 'date' in df.columns:
		df['date'] = pd.to_datetime(df['date'])

	return df

def process_datasets():
    df_prices = pd.read_csv(DATASETS_DIR / "historical_prices.csv")
    df_prices = get_prices(df_prices)
    return df_prices
    

# logic using function/class calls from 1_trading_rules.py
# map stock name, price and date to vectorized dataset
# map after model inference: easier to combine datasets into one df for simulation
# iterate through each row in df and call 1_trading_rules.trading_rules(<df row which is probably formatted as a Series>)to determine buy/sell/hold
def run_simulation():
	"""Run the trading simulation.

	This function uses the available helpers in `trading_rules` to obtain
	predicted impact scores, merges them with historical prices, then
	iterates rows in chronological order executing buy/sell decisions
	returned by `trading_rules.trading_rules(...)`.

	Implementation constraints (per request): modify only this function.
	"""
	# --- load datasets ---
	df_prices = process_datasets()

	# ensure date columns are datetimes for reliable merging/sorting
	if 'date' in df_prices.columns:
		df_prices['date'] = pd.to_datetime(df_prices['date'])

	# get model predictions (returns df with pred_impact_score and symbol,date)
	df_with_preds = trading_rules.model_predict(model_key="tfidf")
	# ensure dates parsed
	if 'date' in df_with_preds.columns:
		df_with_preds['date'] = pd.to_datetime(df_with_preds['date'])

	# after normalization get_prices(), we expect a unified 'price' column
	price_col = 'price'

	# merge predictions with prices on symbol + date
	# Collapse duplicate price rows per (symbol, date) by keeping last occurrence to ensure many-to-one merge
	if 'date' in df_prices.columns:
		df_prices['date'] = pd.to_datetime(df_prices['date'])
	# sort so that drop_duplicates(keep='last') retains the most recent row per (symbol,date)
	df_prices = df_prices.sort_values(['symbol', 'date'])
	df_prices_unique = df_prices.drop_duplicates(subset=['symbol', 'date'], keep='last')

	stock_data = pd.merge(
		df_with_preds,
		df_prices_unique[['symbol', 'date', price_col]],
		on=['symbol', 'date'],
		how='left',
		validate='m:1'
	)

	# sort by date then symbol to simulate chronologically
	stock_data = stock_data.sort_values(['date', 'symbol']).reset_index(drop=True)

	# --- simulation state ---
	initial_balance = 100000.0
	balance = float(initial_balance)
	positions = {}  # symbol -> {'shares': int, 'avg_price': float}
	trade_records = []

	# iterate rows
	for _, row in stock_data.iterrows():
		symbol = row.get('symbol')
		date = row.get('date')
		price = row.get(price_col)
		s = row.get('pred_impact_score', 0)

		# skip if price missing
		if pd.isna(price) or price is None:
			continue

		owned = positions.get(symbol, {}).get('shares', 0)

		# prepare stock dict expected by trading_rules (it reads stock['balance'] inside buy calc)
		stock = {
			'pred_impact_score': float(s) if s is not None else 0.0,
			'price': float(price),
			'balance': float(balance),
		}

		# call trading_rules; it returns an int (shares) or None
		try:
			decision_shares = trading_rules.trading_rules(stock, balance, owned)
		except Exception as e:
			# Surface the exception to help debugging instead of silently
			# swallowing it. Print traceback and the input that caused it.
			import traceback
			print("Exception while calling trading_rules.trading_rules for:", {
				'symbol': symbol,
				'date': date,
				'price': price,
				'pred_impact_score': stock.get('pred_impact_score'),
				'balance': balance,
				'owned': owned,
			})
			traceback.print_exc()
			decision_shares = None

		if decision_shares is None:
			continue

		# determine action by sign of impact score
		if stock['pred_impact_score'] > 0:
			# buy path
			want_shares = int(decision_shares)
			max_affordable = int(balance // stock['price'])
			buy_qty = min(want_shares, max_affordable)
			if buy_qty <= 0:
				continue
			cost = buy_qty * stock['price']
			# update position average price
			prev = positions.get(symbol)
			if prev is None:
				positions[symbol] = {'shares': buy_qty, 'avg_price': stock['price']}
			else:
				total_shares = prev['shares'] + buy_qty
				avg_price = (prev['avg_price'] * prev['shares'] + stock['price'] * buy_qty) / total_shares
				positions[symbol] = {'shares': total_shares, 'avg_price': avg_price}
			balance -= cost
			trade_records.append({
				'Transaction date': pd.to_datetime(date).date().isoformat() if not pd.isna(date) else '',
				'Symbol': symbol,
				'Trade type': 'Buy',
				'Number of shares': buy_qty,
				'Price': float(stock['price']),
				'Transaction amount': -cost,
				'Available cash after the trade': balance,
				'News (headline)': row.get('headline', ''),
				'Impact score': stock['pred_impact_score'],
			})

		elif stock['pred_impact_score'] < 0 and owned > 0:
			# sell path - decision_shares expresses target number to sell per trading_rules implementation
			sell_qty = int(decision_shares)
			# cap to owned
			sell_qty = min(sell_qty, owned)
			if sell_qty <= 0:
				continue
			proceeds = sell_qty * stock['price']
			prev = positions.get(symbol, {'shares': 0, 'avg_price': 0.0})
			gain_loss = sell_qty * (stock['price'] - prev.get('avg_price', 0.0))
			# update position
			remaining = prev['shares'] - sell_qty
			if remaining > 0:
				positions[symbol]['shares'] = remaining
			else:
				positions.pop(symbol, None)
			balance += proceeds
			trade_records.append({
				'Transaction date': pd.to_datetime(date).date().isoformat() if not pd.isna(date) else '',
				'Symbol': symbol,
				'Trade type': 'Sell',
				'Number of shares': sell_qty,
				'Price': float(stock['price']),
				'Transaction amount': proceeds,
				'Available cash after the trade': balance,
				'News (headline)': row.get('headline', ''),
				'Impact score': stock['pred_impact_score'],
				'$gain/loss for the trade': gain_loss,
			})

	# Final liquidation on last available date: sell all remaining positions at last known price
	if len(stock_data) > 0:
		last_date = stock_data['date'].max()
	else:
		last_date = None

	# build price lookup for last_date
	if last_date is not None:
		# ensure df_prices has datetime 'date' and an index we can query
		df_prices['date'] = pd.to_datetime(df_prices['date'])
		price_lookup = df_prices.set_index(['symbol', 'date'])[price_col].to_dict()
		for symbol, pos in list(positions.items()):
			shares = pos['shares']
			# attempt to find price for (symbol, last_date)
			key = (symbol, pd.to_datetime(last_date))
			price = price_lookup.get(key)
			if price is None:
				# try most recent price for symbol
				pf = df_prices[df_prices['symbol'] == symbol].sort_values('date', ascending=False)
				if not pf.empty:
					price = float(pf.iloc[0][price_col])
				else:
					continue
			proceeds = shares * float(price)
			gain_loss = shares * (float(price) - pos.get('avg_price', 0.0))
			balance += proceeds
			trade_records.append({
				'Transaction date': pd.to_datetime(last_date).date().isoformat(),
				'Symbol': symbol,
				'Trade type': 'Sell (liquidation)',
				'Number of shares': shares,
				'Price': float(price),
				'Transaction amount': proceeds,
				'Available cash after the trade': balance,
				'News (headline)': '',
				'Impact score': '',
				'$gain/loss for the trade': gain_loss,
			})

	# write logs
	trades_df = pd.DataFrame(trade_records)
	trades_df.to_csv('trade_log.csv', index=False)

	final_balance = balance
	total_gain_loss = final_balance - initial_balance
	days = (pd.to_datetime(last_date) - pd.to_datetime(stock_data['date'].min())).days if last_date is not None else 0
	years = days / 252 if days > 0 else 0
	annual_return = (final_balance / initial_balance) ** (1 / years) - 1 if years > 0 else 0.0
	total_return_pct = (final_balance / initial_balance - 1.0) * 100.0

	summary = {
		'Total $gain/loss': total_gain_loss,
		'Average annual % return': annual_return * 100.0,
		'Total % return': total_return_pct,
		'Final account balance': final_balance,
	}
	pd.DataFrame([summary]).to_csv('final_summary.csv', index=False)


# make sure to initialize list to keep track of balance
# trade tracking and metric calculations can go in a new df 
# can store data from each trade in df and ouput final to csv or log to file after the end of each trade

# append state columns for state tracking
if __name__ == "__main__":
	run_simulation()
	
