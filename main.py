# This script is used to print the first 10 rows of the historical prices csv
# import pandas as pd
# for chunk in pd.read_csv("datasets/historical_prices.csv", usecols=["symbol"], chunksize=100000):
#     print(chunk["symbol"].unique())

import pandas as pd

path0 = 'datasets/historical_prices.csv.backup'
df0 = pd.read_csv(path0)
print(f"This is the historical prices csv original: \n{df0.head(10)}")

path1 = 'datasets/historical_prices.csv'
df1 = pd.read_csv(path1)
print(f"This is the historical prices csv without AdjClose: \n{df1.head(10)}")

# path2 = 'datasets/historical_prices_with_volatility.csv'
# df2 = pd.read_csv(path2)
# print(f"This is the prices with volatility csv: \n{df2.head(10)}")

# path3 = 'datasets/historical_prices_impact.csv'
# df3 = pd.read_csv(path3)
# print(f"This is the historical prices impact csv: \n{df3.head(10)}")

# path4 = 'datasets/vectorized_news_curated.csv'
# df4 = pd.read_csv(path4)
# print(f"This is the vectorized news curated csv: \n{df4.head(10)}")

# path5 = 'datasets/aggregated_news.csv'
# df5 = pd.read_csv(path5)
# print(f"This is the aggregated news csv: \n{df5.head(10)}")