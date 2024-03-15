import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 从CSV文件中读取数据
df = pd.read_csv('../result_sales_data.csv')

# 将“销售日期”转换为datetime类型，并筛选日期范围
df['销售日期'] = pd.to_datetime(df['销售日期'])
df = df[(df['销售日期'] >= '2023-06-24') & (df['销售日期'] <= '2023-06-30')]

# 获取所有单品类
product_categories = df['单品类'].unique()

# 创建一个空的DataFrame，用于存储预测结果
forecast_df = pd.DataFrame(columns=['日期'] + list(product_categories))

# 对每个单品类进行预测
for category in product_categories:
    print(f"正在处理单品类 {category} ...")

    # 筛选出当前单品类的数据，并按日期聚合
    category_df = df[df['单品类'] == category]
    grouped_df = category_df.groupby('销售日期')['销量(千克)'].sum().reset_index()

    # 设置索引
    grouped_df.set_index('销售日期', inplace=True)

    # 使用所有数据作为训练集
    train = grouped_df.loc['2023-06-24':'2023-06-30']

    # 创建并拟合SARIMA模型（参数需要进一步优化）
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit()

    # 进行预测
    forecast = model_fit.get_forecast(steps=7)
    daily_forecast = forecast.predicted_mean  # 这里已经是一个包含7天预测值的数组

    # 将预测值添加到DataFrame
    forecast_df.loc[:, category] = daily_forecast.tolist()

# 生成日期范围
forecast_df['日期'] = pd.date_range(start='2023-07-01', periods=7)

# 转置DataFrame
forecast_df = forecast_df.transpose()

# 设置第一行为列名
forecast_df.columns = forecast_df.iloc[0]

# 删除第一行，因为它现在成了列名
forecast_df = forecast_df.iloc[1:]

# 重新设置索引
forecast_df.reset_index(inplace=True)

# 重命名索引列
forecast_df.rename(columns={'index': '单品类'}, inplace=True)

# 将预测值存储到CSV文件
forecast_df.to_csv('sales_forecast1.csv', index=False)
