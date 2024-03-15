import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# 从CSV文件中读取数据
df = pd.read_csv('../sales_data0.csv')

# 按照“销售日期”进行分组，并计算每天的总销量
grouped_df = df.groupby('销售日期')['销量(千克)'].sum().reset_index()

# 将“销售日期”转换为datetime类型，并设置为索引
grouped_df['销售日期'] = pd.to_datetime(grouped_df['销售日期'])
grouped_df.set_index('销售日期', inplace=True)

# 明确设置时间序列的频率为'D'（每天）
grouped_df = grouped_df.asfreq('D')

# 进行STL分解
stl = STL(grouped_df['销量(千克)'], seasonal=7)
result = stl.fit()

# 绘制STL分解的结果
fig = plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(result.trend)
plt.title('Trend Component')

plt.subplot(4, 1, 2)
plt.plot(result.seasonal)
plt.title('Seasonal Component')

plt.subplot(4, 1, 3)
plt.plot(result.resid)
plt.title('Residual Component')

plt.subplot(4, 1, 4)
plt.plot(grouped_df['销量(千克)'])
plt.title('Original Series')

plt.tight_layout()
plt.show()
