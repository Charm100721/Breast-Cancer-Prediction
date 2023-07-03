import pandas as pd

data = pd.read_csv("Trades.csv")
df = pd.DataFrame(data)


df = df[df.index > 25138]
df = df[['symbol', 'orderId', 'realizedPnl']]
df = df[~df['symbol'].isin(['AVAXUSDT', 'ADAUSDT', 'LTCUSDT'])]
df = df.drop_duplicates(subset=['orderId'])
df = df[df['realizedPnl'] != 0]
df['win/lose'] = df['realizedPnl'].apply(lambda x: 1 if x > 0 else 0)
print(df)
total_trades = df['realizedPnl'].count()
total_win = df['win/lose'].sum()
winrate = round((total_win/total_trades)*100,2)
print(winrate)