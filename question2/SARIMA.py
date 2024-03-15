import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# 从CSV文件中读取数据
df = pd.read_csv('../result_sales_data.csv')

# 将“销售日期”转换为datetime类型，并筛选日期范围
df['销售日期'] = pd.to_datetime(df['销售日期'])
df = df[(df['销售日期'] >= '2023-06-01') & (df['销售日期'] <= '2023-06-30')]

# 定义大类列表
categories = ['水生根茎类', '花叶类', '花菜类', '茄类', '辣椒类', '食用菌']

# 存储每个类的MAE、RMSE和预测值
mae_dict = {}
rmse_dict = {}
forecast_dict = {}

# 创建一个3x2的总图
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# 对每个大类进行预测
for idx, category in enumerate(categories):
    print(f"正在处理 {category} ...")

    # 筛选出当前大类的数据，并按日期聚合
    category_df = df[df['大类'] == category]
    grouped_df = category_df.groupby('销售日期')['销量(千克)'].sum().reset_index()

    # 设置索引
    grouped_df.set_index('销售日期', inplace=True)

    # 使用所有数据作为训练集
    train = grouped_df.loc['2023-06-01':'2023-06-30']

    # 创建并拟合SARIMA模型（参数需要进一步优化）
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit()

    # 进行预测
    forecast = model_fit.get_forecast(steps=7)
    daily_forecast = forecast.predicted_mean  # 这里已经是一个包含7天预测值的数组

    # 存储预测值
    forecast_dict[category] = daily_forecast.tolist()

    # 计算MAE并存储到字典
    mae = mean_absolute_error(train[-7:], daily_forecast)
    mae_dict[category] = mae

    # 计算RMSE并存储到字典
    rmse = sqrt(mean_squared_error(train[-7:], daily_forecast))
    rmse_dict[category] = rmse

    # 绘制预测结果到子图
    ax = axes[idx // 2, idx % 2]
    ax.plot(train.index[-30:], train[-30:], label='Train')
    ax.plot(daily_forecast.index, daily_forecast, label='Forecast')
    ax.set_title(f"{category} - Sales Forecast for July 2023")
    ax.legend()

    # 旋转横坐标标签以防止重叠
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    # 输出预测和性能指标
    print(f"{category} - 每日预测值: {daily_forecast.tolist()}")
    print(f"{category} - MAE: {mae}")
    print(f"{category} - RMSE: {rmse}")
    print("------")

# 显示总图
plt.tight_layout()
plt.show()

# 初始化一个列表来存储每一天的总预测值
total_daily_forecast = [0] * 7  # 假设我们有7天的预测

# 将每个大类的预测值加到总预测值上
for category, daily_forecast in forecast_dict.items():
    for i in range(len(daily_forecast)):
        total_daily_forecast[i] += daily_forecast[i]

# 输出总预测
print("总预测值（所有大类）:")
print(total_daily_forecast)