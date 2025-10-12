import yfinance
import pandas as pd
import os
import time
from typing import Iterable, Tuple, Dict, Set
import requests 
from time import sleep
from webscraper_utils import is_allowed, polite_get, fetch_article, get_crawl_delay
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

### -----------------------------------------Stock Data Collection ----------------------------------------- ###
def get_symbols_from_news_datasets():
    news_datasets = [
        'news_datasets/analyst_ratings.csv',
        'news_datasets/headlines.csv',
    ]
    symbols = set()
    for dataset in news_datasets:
        if os.path.exists(dataset):
            df = pd.read_csv(dataset)
            if 'symbol' in df.columns:
                symbols.update(df['symbol'].dropna().unique())
            elif 'stock' in df.columns:
                symbols.update(df['stock'].dropna().unique())
    return symbols

def chunked_iterable(iterable: Iterable, size: int):
    items = list(iterable)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _process_download_df(df: pd.DataFrame, batch: Iterable[str], all_data: Dict[str, list], failed_set: Set[str]):
    """Process a yfinance.download DataFrame and populate all_data or failed_set.
    """
    # If no DataFrame was returned (download error), mark whole batch as failed
    if df is None:
        for sym in batch:
            failed_set.add(sym)
        return

    # If df has MultiIndex columns it contains multiple tickers (common case)
    if isinstance(df.columns, pd.MultiIndex):
        for sym in batch:
            # Try to extract symbol-level columns; try level=1 then level=0 as fallback
            try:
                sym_df = df.xs(sym, axis=1, level=1, drop_level=True)
            except Exception:
                try:
                    sym_df = df.xs(sym, axis=1, level=0, drop_level=True)
                except Exception:
                    # If neither works, produce an empty DataFrame for this symbol
                    sym_df = pd.DataFrame()

            # If there's no data for this symbol, mark it as failed and continue
            if sym_df.empty:
                failed_set.add(sym)
                continue

            # Prefer 'Adj Close' if available, otherwise use 'Close' as fallback
            adj_col = 'Adj Close' if 'Adj Close' in sym_df.columns else 'Close'
            rows = []
            # Build per-date row dicts for this symbol
            for idx, r in sym_df.iterrows():
                rows.append({
                    'date': idx.date().isoformat(),
                    'symbol': 's&p' if sym == '^GSPC' else sym,
                    'open': float(r.get('Open', float('nan'))),
                    'high': float(r.get('High', float('nan'))),
                    'low': float(r.get('Low', float('nan'))),
                    'close': float(r.get(adj_col, float('nan'))),
                    'volume': int(r.get('Volume', 0) if not pd.isna(r.get('Volume', None)) else 0)
                })
            all_data[sym] = rows
    else:
        # Non-MultiIndex layout: usually a single-symbol batch or an unexpected layout
        if len(batch) == 1:
            # Single-symbol batch: df itself is that symbol's DataFrame
            sym = batch[0]
            sym_df = df
            if sym_df.empty:
                failed_set.add(sym)
            else:
                adj_col = 'Adj Close' if 'Adj Close' in sym_df.columns else 'Close'
                rows = []
                for idx, r in sym_df.iterrows():
                    rows.append({
                        'date': idx.date().isoformat(),
                        'symbol': 's&p' if sym == '^GSPC' else sym,
                        'open': float(r.get('Open', float('nan'))),
                        'high': float(r.get('High', float('nan'))),
                        'low': float(r.get('Low', float('nan'))),
                        'close': float(r.get(adj_col, float('nan'))),
                        'volume': int(r.get('Volume', 0) if not pd.isna(r.get('Volume', None)) else 0)
                    })
                all_data[sym] = rows
        else:
            # Fallback: multiple symbols but no MultiIndex; try to match columns by substring
            for sym in batch:
                matches = [c for c in df.columns.astype(str) if sym in str(c)]
                # If no matching columns found, mark as failed
                if not matches:
                    failed_set.add(sym)
                    continue
                sym_df = df.loc[:, matches]
                rows = []
                # Use positional access as a last-resort if header format is unusual
                for idx, r in sym_df.iterrows():
                    rows.append({
                        'date': idx.date().isoformat(),
                        'symbol': 's&p' if sym == '^GSPC' else sym,
                        'open': float(r.iloc[0]) if len(r) > 0 else float('nan'),
                        'high': float(r.iloc[1]) if len(r) > 1 else float('nan'),
                        'low': float(r.iloc[2]) if len(r) > 2 else float('nan'),
                        'close': float(r.iloc[3]) if len(r) > 3 else float('nan'),
                        'volume': int(r.iloc[4]) if len(r) > 4 and not pd.isna(r.iloc[4]) else 0
                    })
                all_data[sym] = rows


def get_stock_data(symbols: Iterable[str], start_date: str, end_date: str,
                   csv_path: str = "datasets/historical_prices.csv",
                   batch_size: int = 40,
                   delay_between_batches: float = 1.0) -> Tuple[Dict[str, list], Set[str]]:
    """Download historical data in batches without per-batch retries.
    """
    symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
    all_data: Dict[str, list] = {}
    failed_stage1: Set[str] = set()

    # First pass: single attempt per batch
    for batch in chunked_iterable(symbols, batch_size):
        try:
            df = yfinance.download(batch, start=start_date, end=end_date, interval='1d', threads=False, progress=False)
        except Exception as e:
            print(f"Batch download exception for {batch[:5]}...: {e}. Marking batch as failed.")
            failed_stage1.update(batch)
            time.sleep(delay_between_batches)
            continue

        if df is None or df.empty:
            print(f"No data returned for batch: {batch}")
            failed_stage1.update(batch)
            time.sleep(delay_between_batches)
            continue

        # Process returned DataFrame (use helper to avoid duplicated logic)
        _process_download_df(df, batch, all_data, failed_stage1)

        time.sleep(delay_between_batches)

    failed_stage1_list = sorted(list(failed_stage1))

    # Final retry pass: one attempt for all failed symbols (batched)
    remaining_failed: Set[str] = set()
    if failed_stage1_list:
        print("Starting final retry pass for failed symbols...")
        for batch in chunked_iterable(failed_stage1_list, batch_size):
            try:
                df = yfinance.download(batch, start=start_date, end=end_date, interval='1d', threads=False, progress=False)
            except Exception as e:
                print(f"Final retry exception for batch {batch[:5]}...: {e}. Marking batch as still failed.")
                remaining_failed.update(batch)
                time.sleep(delay_between_batches)
                continue

            if df is None or df.empty:
                print(f"No data on final retry for batch: {batch}")
                remaining_failed.update(batch)
                time.sleep(delay_between_batches)
                continue

            # Process successes from final retry using helper; collect remaining failures separately
            # Use a temporary set to capture failures in this pass
            temp_failed: Set[str] = set()
            _process_download_df(df, batch, all_data, temp_failed)
            # Any symbol that ended up in temp_failed should be considered still failed
            remaining_failed.update(temp_failed)

            time.sleep(delay_between_batches)

    # Save flattened CSV with the required columns (as before)
    flat_rows = []
    for sym, rows in all_data.items():
        flat_rows.extend(rows)

    if flat_rows:
        out_dir = os.path.dirname(csv_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        df_out = pd.DataFrame(flat_rows, columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
        df_out.sort_values(['symbol', 'date'], inplace=True)
        df_out.to_csv(csv_path, index=False)
        print(f"Saved historical data to {csv_path} ({df_out.shape[0]} rows, {df_out['symbol'].nunique()} symbols)")
    else:
        print("No historical rows to save.")

### -----------------------------------------Web Scrapping ----------------------------------------- ###

def get_news_data():
    """
    Collect news article text from supplied news datasets and save into a
    single CSV file (`datasets/all_news.csv`).
    """

    news_datasets = [
        'news_datasets/analyst_ratings.csv',
        'news_datasets/headlines.csv',
    ]

    all_news_data = pd.DataFrame()
    for dataset in news_datasets:
        df = pd.read_csv(dataset)
        all_news_data = pd.concat([all_news_data, df], ignore_index=True)
    # HTTP session reused across requests for connection pooling
    session = requests.Session()
    # Simple in-memory cache mapping URL text to avoid duplicate calls for the same URL within a single run.
    cache: Dict[str, str] = {}
    cache_lock = threading.Lock()
    stop_event = threading.Event()

    # Ensure datasets dir exists
    out_dir = os.path.join('datasets')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # CSV-based resume: load existing URLs from datasets/all_news.csv 
    csv_path = os.path.join(out_dir, 'all_news.csv')
    csv_exists = os.path.exists(csv_path)
    processed_urls = set()
    next_id = 1
    # Keep a mapping of already-seen URL
    processed_map: Dict[str, str] = {}
    cols = ['id', 'headline', 'URL', 'article', 'publisher', 'date', 'symbol']
    if csv_exists:
        #Resume where it left off
        try:
            existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
            # If existing columns don't match sample, back up the old file and start a fresh CSV
            if set(existing_cols) != set(cols):
                backup_path = csv_path + '.bak'
                print(f"Existing CSV columns differ from sample; backing up old CSV to {backup_path} and starting a new one.")
                os.replace(csv_path, backup_path)
                csv_exists = False
            else:
                # Read in chunks the URL and id columns to build processed set and find max id
                max_id = 0
                for chunk in pd.read_csv(csv_path, usecols=['URL', 'id'], dtype=str, chunksize=100_000):
                    processed_urls.update(chunk['URL'].dropna().astype(str).tolist())
                    if 'id' in chunk.columns:
                        ids = pd.to_numeric(chunk['id'], errors='coerce').dropna()
                        if not ids.empty:
                            max_id = max(max_id, int(ids.max()))
                next_id = max_id + 1
                # build URL article map by reading article column in chunks
                for chunk in pd.read_csv(csv_path, usecols=['URL', 'article'], dtype=str, chunksize=100_000):
                    for u, a in zip(chunk['URL'].astype(str).tolist(), chunk['article'].astype(str).tolist()):
                        if u and u not in processed_map:
                            processed_map[u] = a
            
        except Exception as e:
            print(f"Warning: failed to read existing CSV index: {e}. Continuing with empty processed set.")
            processed_urls = set()
            csv_exists = False

    #Buffer size
    flush_every = 100
    buffer = []

    
    symbol_cols = ['symbol', 'stock']

    def pick_first(r, candidates):
        for c in candidates:
            if c in r and pd.notna(r[c]):
                return r[c]
        return ""

    def process_row(row):
        # Locate a URL (support common variants)
        url = None
        for c in ('url', 'URL', 'link', 'Link'):
            if c in row and pd.notna(row[c]):
                url = str(row[c])
                break
        if not url:
            return None  # skip
        # Respect robots.txt
        if not is_allowed(url):
            print(f"Skipping (disallowed by robots.txt): {url}")
            return None
        date_val = row['date']
        symbol_val = pick_first(row, symbol_cols)
        headline_val = row['headline']
        publisher_val = row['publisher']
        # Fetch article text: consult cache first
        with cache_lock:
            article_text = cache.get(url)
        if article_text is None:
            if stop_event.is_set():
                return None
            # Perform a single polite GET using the shared session, then
            resp = polite_get(url, session=session, default_delay=1.0)
            if resp is None:
                article_text = ""
            else:
                article_text = fetch_article(url, resp=resp, session=session) or resp.text or ""
            with cache_lock:
                cache[url] = article_text
        
        row_out = {
            'id': None,  # will be set later
            'headline': str(headline_val) if headline_val is not None else "",
            'URL': url,
            'article': article_text,
            'publisher': str(publisher_val) if publisher_val is not None else "",
            'date': str(date_val) if date_val is not None else "",
            'symbol': str(symbol_val) if symbol_val is not None else "",
        }
        return (url, row_out)

    try:
        
        # Build a deduplicated list of rows to process, mark scheduled URLs
        rows_to_process = []
        scheduled_urls = set()
        for _, row in all_news_data.iterrows():
            # Find a URL value in the row using the same column candidates
            url = None
            for c in ('url', 'URL', 'link', 'Link'):
                if c in row and pd.notna(row[c]):
                    url = str(row[c])
                    break
            if not url or url in scheduled_urls:
                continue
            # If we've already processed this URL previously, duplicate the stored article text
            if url in processed_urls:
                date_val = row['date']
                symbol_val = pick_first(row, symbol_cols)
                headline_val = row['headline']
                publisher_val = row['publisher']
                article_text = processed_map.get(url, "")
                row_out = {
                    'id': None,
                    'headline': str(headline_val) if headline_val is not None else "",
                    'URL': url,
                    'article': article_text,
                    'publisher': str(publisher_val) if publisher_val is not None else "",
                    'date': str(date_val) if date_val is not None else "",
                    'symbol': str(symbol_val) if symbol_val is not None else "",
                }
                buffer.append(row_out)
                scheduled_urls.add(url)
                # If buffer reached threshold, flush now to avoid waiting for threads
                if len(buffer) >= flush_every:
                    try:
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir, exist_ok=True)
                        for i, r in enumerate(buffer):
                            r['id'] = next_id + i
                        df_flush = pd.DataFrame(buffer, columns=cols)
                        df_flush.to_csv(csv_path, mode='a', header=not csv_exists, index=False, encoding='utf-8')
                    except Exception as e:
                        print(f"Failed to flush buffer to CSV during pre-buffering: {e}")
                    else:
                        for r in buffer:
                            processed_urls.add(r['URL'])
                        next_id += len(buffer)
                        buffer = []
                        csv_exists = True
                continue
            scheduled_urls.add(url)
            rows_to_process.append(row)

        max_workers = 8
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = []
        try:
            for row in rows_to_process:
                if stop_event.is_set():
                    break
                futures.append(executor.submit(process_row, row))

            for future in as_completed(futures):
                if stop_event.is_set():
                    break
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Task error: {e}")
                    continue
                if result is None:
                    continue
                url, row_out = result
                buffer.append(row_out)
                processed_urls.add(url)
                if len(buffer) >= flush_every:
                    try:
                        for i, r in enumerate(buffer):
                            r['id'] = next_id + i
                        df_flush = pd.DataFrame(buffer, columns=cols)
                        df_flush.to_csv(csv_path, mode='a', header=not csv_exists, index=False, encoding='utf-8')
                    except Exception as e:
                        print(f"Failed to flush buffer to CSV: {e}")
                    else:
                        for r in buffer:
                            processed_urls.add(r['URL'])
                        next_id += len(buffer)
                        buffer = []
                        csv_exists = True
        except KeyboardInterrupt:
            print('\nInterrupted by user - cancelling pending tasks...')
            stop_event.set()
            # Cancel futures that haven't begun
            for f in futures:
                f.cancel()
        finally:
            executor.shutdown(wait=False)
    except KeyboardInterrupt:
        print('\nInterrupted by user - flushing buffer before exit...')
    finally:
        if buffer:
            try:
                for i, r in enumerate(buffer):
                    r['id'] = next_id + i
                df_flush = pd.DataFrame(buffer, columns=cols)
                df_flush.to_csv(csv_path, mode='a', header=not csv_exists, index=False, encoding='utf-8')
            except Exception as e:
                print(f"Failed to flush final buffer to CSV: {e}")
            else:
                for r in buffer:
                    processed_urls.add(r['URL'])
                next_id += len(buffer)
                buffer = []
        
    

if __name__ == "__main__":
    #symbols = get_symbols_from_news_datasets()
    #get_stock_data(symbols, start_date="2009-01-01", end_date="2020-12-31")
    print("WEB SCRAPING")
    get_news_data()
    
