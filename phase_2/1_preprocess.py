

import pandas as pd




def preprocess(csv_path: str,start_date: str):
    df = pd.read_csv(csv_path, low_memory=False)

    if 'date' not in df.columns:
        raise ValueError(f"Date column not found in {csv_path}")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).copy() 
    df = df.sort_values(by=['date'], ascending=True).reset_index(drop=True)

    # Normalize start_date and compute end_date
    start_dt = pd.to_datetime(start_date)
    if pd.isna(start_dt):
        raise ValueError(f'Could not parse start_date: {start_date}')
    end_dt = start_dt + pd.DateOffset(years=3)
    mask = (df['date'] >= start_dt) & (df['date'] < end_dt)
    slice_df = df.loc[mask].copy().reset_index(drop=True)

    if slice_df.empty:
        raise ValueError(f'No rows found in {csv_path} between {start_dt.date()} and {end_dt.date()}')

    # Chronological split: first `train_frac` portion is training
    n = len(slice_df)
    n_train = int(round(n * 0.8))
    # Ensure at least one row in train and test if possible
    if n_train == 0 and n > 1:
        n_train = 1
    if n_train >= n:
        n_train = n - 1 if n > 1 else n

    train_df = slice_df.iloc[:n_train].reset_index(drop=True)
    test_df = slice_df.iloc[n_train:].reset_index(drop=True)

    return train_df, test_df

if __name__ == '__main__':
    print("Starting")
    print(preprocess('datasets/vectorized_news_dtm.csv', "2009-1-1"))
