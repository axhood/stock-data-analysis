from itertools import groupby
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from test1 import vol_mean, result

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据，把 date 转为日期格式
df = pd.read_csv('./data/stock_data2.csv')
df['date'] = pd.to_datetime(df['date'])
print(df)

# 计算每只股票的收益率（pct_change）
df = df.sort_values(['code', 'date'], ignore_index=True)
df['return'] =df.groupby('code')['close'].pct_change()
print(df)

# 计算5 日均线、20 日均线
print('5日均线:')
df['ma5'] = df.groupby('code')['close'].rolling(5,min_periods=1).mean().values
print('20日均线:')
df['ma20'] = df.groupby('code')['close'].rolling(20,min_periods=1).mean().values
print(df)
# 筛选放量上涨的交易日

def hot_day(data):
    vol_mean = data['volume'].mean()
    return  data[(data['volume']>vol_mean*1.5) &
                  (data['close']>data['open'])]
result = df.groupby('code').apply(hot_day)
print('放量上涨天数',len(result))



# 绘制 3 张基础图：
# 收盘价折线图
# 成交量柱状图
# 收益率直方图
stock = df[df['code'] == 600001]
plt.figure(figsize = (10,10))
plt.plot(stock['date'],stock['close'],label='收盘价')
plt.plot(stock['date'],stock['ma5'],label='ma5')
plt.plot(stock['date'],stock['ma20'],label='ma20')
plt.title('600001 收盘价与均线')
plt.legend()
plt.grid(alpha=0.3)

plt.figure(figsize = (10,8))
plt.bar(stock['date'],stock['volume'],color='yellow')
plt.title('成交量')
plt.grid(alpha=0.3)

plt.figure(figsize = (8,5))
plt.hist(stock['return'].dropna(),bins=8,color='red',alpha=0.7)
plt.title('收益率分布')
plt.grid(alpha=0.3)

plt.show()
