import pandas as pd 
import os

# news_list=[['2020-01-01', 'www.abc.com', 'Hello', 'World'],['2020-05-25', 'www.apple.com', 'Bye', 'Python']]

# for i in news_list:
#     i.insert(1, "AAPL")

df = pd.read_csv('/Users/jeongminsol/stock_psychic_api/com_stock_api/yhnews/data/AAPL_sentiment.csv')

df["Ticker"] = "AAPL"
df_reorder = df[['Date', 'Ticker', 'Link', 'Headline', 'neg', 'neu', 'pos', 'compound']] # rearrange column here

df_reorder.to_csv('/Users/jeongminsol/stock_psychic_api/com_stock_api/yhnews/data/AAPL_sentiment2.csv', index=False)
