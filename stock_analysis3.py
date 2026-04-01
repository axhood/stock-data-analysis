import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import pyplot as plt
from numpy.ma.core import mean
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 累计收益 + 回撤 + 最大回撤
# 计算 ret_cum（累计收益，初始 1）
# 计算 roll_max（历史最高累计收益）
# 计算 drawdown（每期回撤）
# 求 最大回撤（MDD） 及发生日期
df = pd.read_csv('./data/stock_data4.csv',parse_dates=['date'],index_col='date')
print(df.head())
df['ret_cum'] = (df['return']+1).cumprod()
df['roll_max'] = df['ret_cum'].cummax()
df['drawdown'] = df['ret_cum']/df['roll_max']-1
mdd = df['drawdown'].min()
mdd_date = df['drawdown'].idxmin()
print('最大回撤为{:.2%}'.format(mdd))
print('日期为',mdd_date)



# 收益率统计
# 计算 年化收益率（252 交易日）
# 计算 日波动率、年化波动率
# 计算 胜率（上涨天数占比）、平均日收益、最大单日涨幅 / 跌幅
mean_ret = df['return'].mean()
std_ret = df['return'].std()
annual_ret = mean_ret*252
annual_vol = std_ret*np.sqrt(252)
win_rate = (df['return']>0).mean()
max_up = df['return'].max()
max_down = df['return'].min()
print('日波动率为{:.2%}'.format(std_ret))
print('年波动率为{:.2%}'.format(annual_vol))
print('年化收益率为{:.2%}'.format(annual_ret))
print('平均日收益为{:.2%}'.format(mean_ret))
print('胜率为{:.2%}'.format(win_rate))
print('最大单日涨幅为{:.2%}'.format(max_up))
print('最大单日跌幅为{:.2%}'.format(max_down))


# 技术指标 + 策略信号
# 计算 5 日 / 20 日均线（ma5, ma20）
# 生成金叉 / 死叉信号：ma5 上穿 ma20 = 买入信号，下穿 = 卖出信号
# 按信号回测：持有至反向信号，计算策略收益率、最大回撤
df['ma5'] = df['return'].rolling(5).mean()
df['ma20'] = df['return'].rolling(20).mean()
df['gold'] = df['ma5'] > df['ma20']
df['signal'] = df['gold'].astype(int).diff()
df['position'] = df['gold'].astype(int)
df['strategy_ret'] = df['position'].shift(1)*df['return']
df['strategy_cum'] = (1+df['strategy_ret']).cumprod()
df['strategy_rollmax'] = df['strategy_cum'].cumsum()
df['strategy_dd'] = df['strategy_cum']/df['strategy_rollmax']-1
strategy_mdd = df['strategy_dd'].min()
print('策略最终净值为{:.2%}'.format(df['strategy_cum'].iloc[-1]))
print('策略最大回撤为{:.2%}'.format(strategy_mdd))


# 2×1 子图：
# 上图：收盘价 + 5/20 日均线
# 下图：回撤曲线（红色填充）
# 标注 最大回撤点
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,10))
ax1.plot(df.index,df['close'],label='收盘价',linewidth=2)
ax1.plot(df.index,df['ma5'],label='ma5',linewidth=2)
ax1.plot(df.index,df['ma20'],label='ma20',linewidth=2)
ax1.set_title('收盘价与均线')
ax1.legend()
ax1.grid(alpha=0.4)
ax2.fill_between(df.index,df['drawdown'],0,color='red',alpha=0.5)
ax2.plot(df.index,df['drawdown'],label='回撤曲线',color='red',linewidth=2)
ax2.set_title('回撤曲线')
ax2.legend()
ax2.grid(alpha=0.4)
plt.tight_layout()
plt.show()
