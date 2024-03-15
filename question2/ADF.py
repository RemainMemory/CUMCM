import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 从CSV文件中读取数据
df = pd.read_csv('../sales_data0.csv')

# 按照“销售日期”进行分组，并计算每天的总销量
grouped_df = df.groupby('销售日期')['销量(千克)'].sum().reset_index()

# 进行ADF测试
result = adfuller(grouped_df['销量(千克)'])

# 输出ADF测试结果
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# 根据p-value判断数据是否平稳
if result[1] <= 0.05:
    print('数据是平稳的')
else:
    print('数据是非平稳的')
