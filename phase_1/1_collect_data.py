import yfinance
import pandas as pd
import os
import time
from typing import Iterable, Tuple, Dict, Set
import requests 
from webscraper_utils import is_allowed, polite_get, fetch_article
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

### -----------------------------------------Stock Data Collection ----------------------------------------- ###
def get_symbols_from_news_datasets():
    news_datasets = [
        'news_datasets/analyst_ratings.csv',
        'news_datasets/headlines.csv',
    ]
    return get_symbols_from_news_datasets_filtered(None)


def normalize_symbol(sym: str) -> str:
    """Normalize symbol text to a canonical form used for comparisons/Yahoo format.
    Examples: 'brk.b' -> 'BRK-B'
    """
    if not isinstance(sym, str):
        return ""
    return sym.strip().upper().replace('.', '-')


def get_symbols_from_news_datasets_filtered(allowed_symbols: Set[str] = None) -> Set[str]:
    """Return distinct symbols found in the news datasets.

    If allowed_symbols is provided (set of normalized symbols), only returns
    symbols that are present in that allowed set.
    """
    news_datasets = [
        'news_datasets/analyst_ratings.csv',
        'news_datasets/headlines.csv',
    ]
    symbols = set()
    allowed_norm = None
    if allowed_symbols is not None:
        allowed_norm = set(normalize_symbol(s) for s in allowed_symbols)

    for dataset in news_datasets:
        if os.path.exists(dataset):
            df = pd.read_csv(dataset)
            # support columns 'symbol' or 'stock'
            col = None
            if 'symbol' in df.columns:
                col = 'symbol'
            elif 'stock' in df.columns:
                col = 'stock'
            if col:
                for s in df[col].dropna().astype(str).unique():
                    s_norm = normalize_symbol(s)
                    if allowed_norm is None or s_norm in allowed_norm:
                        symbols.add(s_norm)
    return symbols


def get_sp500_symbols() -> Set[str]:
    """Attempt to obtain the S&P 500 constituent symbols.

    """
    sources_tried = []
    # 1) Yahoo Finance components page
    try:
        url = 'https://finance.yahoo.com/quote/%5EGSPC/components'
        tables = pd.read_html(url)
        if tables:
            # find a table that contains a Symbol-like column
            for t in tables:
                cols = [c.lower() for c in t.columns.astype(str).tolist()]
                if any('symbol' in c for c in cols):
                    sym_col = [c for c in t.columns if 'symbol' in str(c).lower()][0]
                    syms = t[sym_col].astype(str).tolist()
                    return set(normalize_symbol(s) for s in syms if s and str(s).strip())
        sources_tried.append('yahoo')
    except Exception:
        sources_tried.append('yahoo-failed')

    # 2) DataHub mirror (raw CSV)
    try:
        url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
        df = pd.read_csv(url)
        if 'Symbol' in df.columns:
            return set(normalize_symbol(s) for s in df['Symbol'].astype(str).tolist())
        sources_tried.append('datahub')
    except Exception:
        sources_tried.append('datahub-failed')

    # 3) Wikipedia fallback (previous behaviour)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        if tables:
            sp500 = tables[0]
            if 'Symbol' in sp500.columns:
                return set(normalize_symbol(s) for s in sp500['Symbol'].astype(str).tolist())
        sources_tried.append('wikipedia')
    except Exception:
        sources_tried.append('wikipedia-failed')

    print(f"Failed to fetch S&P500 symbols from known sources ({sources_tried}). Returning empty set.")
    return set()

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

def get_news_data(allowed_symbols: Set[str] = None, start_date: str = None, end_date: str = None):
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
    # Normalize allowed_symbols for quick membership checks
    allowed_norm = None
    if allowed_symbols is not None:
        allowed_norm = set(normalize_symbol(s) for s in allowed_symbols)

    # Parse date bounds and convert to integer nanosecond epoch to avoid tz-naive vs tz-aware comparison errors
    def _ts_to_ns(x):
        try:
            t = pd.to_datetime(x, errors='coerce', utc=True)
            if pd.isna(t):
                return None
            return int(t.value)
        except Exception:
            return None

    start_ns = _ts_to_ns(start_date) if start_date is not None else None
    end_ns = _ts_to_ns(end_date) if end_date is not None else None

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
        # Filter by symbol and date (if requested)
        symbol_val = pick_first(row, symbol_cols)
        symbol_norm = normalize_symbol(symbol_val)
        if allowed_norm is not None and symbol_norm not in allowed_norm:
            return None
        # parse date
        date_raw = row.get('date', None)
        # Convert row date to integer ns for robust comparison
        date_ns = _ts_to_ns(date_raw)
        if start_ns is not None and (date_ns is None or date_ns < start_ns):
            return None
        if end_ns is not None and (date_ns is None or date_ns >= end_ns):
            return None
        date_val = row['date']
        headline_val = row.get('headline', None)
        publisher_val = row.get('publisher', None)
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
                # fetch_article now returns None when the page is promotional/boilerplate
                fetched = fetch_article(url, resp=resp, session=session)
                if fetched is None:
                    # Detected as promotional / boilerplate (e.g., Benzinga Pro ad) -> skip
                    return None
                # Use fetched article text if available, otherwise fall back to full resp.text
                article_text = fetched or resp.text or ""
            with cache_lock:
                cache[url] = article_text
        
        row_out = {
            'id': None,  # will be set later
            'headline': str(headline_val) if headline_val is not None else "",
            'URL': url,
            'article': article_text,
            'publisher': str(publisher_val) if publisher_val is not None else "",
            'date': str(date_val) if date_val is not None else "",
            'symbol': symbol_norm if symbol_norm is not None else "",
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
                # Enforce same symbol/date filtering as for new rows
                symbol_val = pick_first(row, symbol_cols)
                symbol_norm = normalize_symbol(symbol_val)
                if allowed_norm is not None and symbol_norm not in allowed_norm:
                    continue
                # date filter
                date_ns = _ts_to_ns(row.get('date', None))
                if start_ns is not None and (date_ns is None or date_ns < start_ns):
                    continue
                if end_ns is not None and (date_ns is None or date_ns >= end_ns):
                    continue
                date_val = row['date']
                headline_val = row.get('headline', None)
                publisher_val = row.get('publisher', None)
                article_text = processed_map.get(url, "")
                row_out = {
                    'id': None,
                    'headline': str(headline_val) if headline_val is not None else "",
                    'URL': url,
                    'article': article_text,
                    'publisher': str(publisher_val) if publisher_val is not None else "",
                    'date': str(date_val) if date_val is not None else "",
                    'symbol': symbol_norm,
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
    # Desired flow:
    # 1) obtain S&P500 set, 2) obtain dataset symbols, 3) intersect and pick top-500,
    # 4) download stock data only for that intersection, 5) fetch news only for those symbols.
    start = "2009-01-01"
    end = "2015-01-01"

    # 1) obtain S&P500 set
    sp500_set = get_sp500_symbols()

    # 2) obtain symbols mentioned in the news datasets
    news_symbols = get_symbols_from_news_datasets()

    # 3) intersect (only keep symbols that are in both sets). If we failed to get
    # the S&P500 set, fall back to taking up to 500 news-derived symbols.
    if sp500_set:
        allowed_set = set(s for s in news_symbols if s in sp500_set)
    else:
        print("Warning: could not fetch authoritative S&P500 list; defaulting to news-derived symbols (first 500).")
        allowed_set = set(list(news_symbols)[:500])

    # Limit to 500 (in case intersection is larger)
    allowed_list = sorted(list(allowed_set))[:500]

    if not allowed_list:
        print("No overlapping symbols between news datasets and S&P500 (or no symbols available). Exiting.")
    else:
        print(f"Proceeding with {len(allowed_list)} symbols (intersection with S&P500). Downloading historical data from {start} to {end}...")
        get_stock_data(allowed_list, start_date=start, end_date=end, csv_path="datasets/historical_prices.csv")

        # Finally, fetch news only for the allowed symbols and date window
        print("WEB SCRAPING (news collection for selected symbols)")
        get_news_data(allowed_symbols=set(allowed_list), start_date=start, end_date=end)
    
