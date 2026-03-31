import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import figure


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 读取 CSV，将 date 转为 datetime 并设为索引。
# 计算日收益率
# 剔除第一行缺失值
df = pd.read_csv('./data/stock_data3.csv')
print(df.head())
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df['return'] = df['close'].pct_change()
print(df.head())
df = df.dropna()
print("=" * 50)
print("任务1：数据处理完成，前5行数据：")
print(df.head())

# 技术指标
# 手动实现 5 日移动均线 MA5（不要用 talib）。
# 手动实现 20 日移动均线 MA20。
# 计算 价格偏离率：
# 找出 deviation 绝对值 > 2% 的日期，统计天数

df['ma5'] = df['close'].rolling(5).mean()
df['ma20'] = df['close'].rolling(20).mean()
df['deviation'] = (df['close']-df['ma5'])/df['ma5']
print(df)
df_dev = df[df['deviation'].abs()>0.02].copy()
dev_days = len(df_dev)
print(df_dev[['close','ma5','deviation']].round(4).to_string())
print('deviation 绝对值 > 2% 的天数',dev_days)



# 波动率与风险指标
# 计算年化波动率（交易日 252 天）：
# 计算滚动 20 日波动率（rolling std）。
# 画出：收盘价 + 滚动 20 日波动率（双轴图）
daily_vol = df['return'].std()
annual_vol = daily_vol*np.sqrt(252)
print(annual_vol.round(4))
df['vol_20'] = df['return'].rolling(20).std().round(4)
# fig, ax1 = plt.subplots(figsize=(18,10))
# ax1.plot(df.index, df['close'],color='blue',linewidth=1.8,label='收盘价')
# ax1.set_ylabel('收盘价',color='blue',fontsize=22)
# ax1.tick_params(axis='y', labelcolor='blue')
# ax2=ax1.twinx()
# ax2.plot(df.index,df['vol_20'],color='red',linewidth=1.6,label='滚动 20 日波动率')
# ax2.set_ylabel('滚动 20 日波动率',color='red',fontsize=22)
# ax2.tick_params(axis='y', labelcolor='red')
# plt.title('收盘价 + 滚动 20 日波动率（双轴图）',fontsize=26)
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# 量价关系
# 计算 量价相关系数（volume 与 daily return）。
# 画出 散点图：收益率 vs 成交量，加回归线。
# 回答：量价正相关还是负相关？强弱如何？
corr = df['return'].corr(df['volume'])
print('量价相关系数为',corr.round(4))
x = df['volume']
y = df['return']
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
# plt.figure(figsize=(12, 6))
# plt.scatter(x, y, s=20, alpha=0.6, color='#5D737E')
# plt.plot(x, p(x), "r--", linewidth=1.5, label=f'拟合线 y={z[0]:.2e}x+{z[1]:.4f}')
# plt.xlabel('成交量')
# plt.ylabel('日收益率')
# plt.title(f'量价散点图 | 相关系数={corr:.4f}', fontsize=20)
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()



# 均线策略信号
# 金叉：MA5 上穿 MA20 → 信号 = 1
# 死叉：MA5 下穿 MA20 → 信号 =-1
# 统计全段金叉、死叉各出现几次
# 画出 MA5、MA20、Close 三线图

df['ma_diff'] = df['ma5'] - df['ma20']
df['signal'] = 0
df.loc[(df['ma_diff'] > 0) & (df['ma_diff'].shift(1) <= 0), 'signal'] = 1  # 金叉
df.loc[(df['ma_diff'] < 0) & (df['ma_diff'].shift(1) >= 0), 'signal'] = -1  # 死叉
golden = df[df['signal'] == 1].shape[0]
dead = df[df['signal'] == -1].shape[0]
print('金叉次数',golden)
print('死叉次数',dead)
# plt.figure(figsize=[14,16])
# plt.plot(df['ma5'], label='ma5',linewidth=1.8)
# plt.plot(df['ma20'], label='ma20',linewidth=1.8)
# plt.plot(df['close'], label='收盘价',linewidth=2)
# plt.scatter(df[df['signal']==1].index, df[df['signal']==1]['ma5'], marker='^', color='red', s=60, label='金叉')
# plt.scatter(df[df['signal']==-1].index, df[df['signal']==-1]['ma5'], marker='v', color='green', s=60, label='死叉')
# plt.title('均线金叉死叉', fontsize=14)
# plt.legend()
# plt.grid(alpha=0.4)
# plt.tight_layout()
# plt.show()



# 任务 6：最大回撤
# 计算累计收益曲线：cumprod(1 + return)
# 计算滚动最大值（高位线）。
# 画出：累计收益 + 回撤曲线（上下子图)
df['ret_cum'] = (1+df['return']).cumprod()
df['roll_max'] = df['ret_cum'].cummax()
df['drawdown'] = df['ret_cum']/df['roll_max']-1
max_drawdown = df['drawdown'].min()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.plot(df.index, df['ret_cum'], color='#2E86AB', linewidth=2, label='累计收益')
ax1.set_title('累计收益曲线', fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)
ax2.fill_between(df.index, df['drawdown'], 0, color='#F24236', alpha=0.5, label='回撤')
ax2.plot(df.index, df['drawdown'], color='#F24236', linewidth=1.5)
ax2.set_ylabel('回撤', fontsize=12)
ax2.set_title('回撤曲线', fontsize=14)
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()
