import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 读取第一个CSV文件并只保留2023年的数据
df = pd.read_csv('../sales_data0.csv', low_memory=False)
df['销售日期'] = pd.to_datetime(df['销售日期'])
df = df[df['销售日期'].dt.year == 2023]

# 计算实际成本
df['成本'] = df['批发价格'] * (1 + df['单品损耗率'] / 100)

# 读取第二个CSV文件（7月的预测数据）
future_df = pd.read_csv('sales_forecast0_adjust.csv')

# 将数据融化成长格式
future_df_melted = pd.melt(future_df, id_vars=['单品类', '批发价格', '单品损耗率'], var_name='日期',
                           value_name='销量(千克)')

# 计算预计成本
future_df_melted['成本'] = future_df_melted['批发价格'] * (1 + future_df_melted['单品损耗率'] / 100)

# ...（其他代码不变）

# 初始化一个空的DataFrame来存储预测结果
predictions = pd.DataFrame()

# 按照不同的单品类分组数据
for category, group in df.groupby('单品类'):
    # 训练模型
    X = group[['成本', '销量(千克)']]
    y = group['销售单价(元/千克)']
    model = LinearRegression()
    model.fit(X, y)

    # 使用模型进行预测
    future_group = future_df_melted[future_df_melted['单品类'] == category].copy()  # 注意这里使用了.copy()
    future_X = future_group[['成本', '销量(千克)']]
    if not future_X.empty:  # 检查是否有数据
        future_group.loc[:, '预测销售单价(元/千克)'] = model.predict(future_X)  # 使用.loc[]进行赋值

        # 将预测结果添加到predictions DataFrame中
        predictions = pd.concat([predictions, future_group])

# 输出或保存预测结果
print(predictions[['单品类', '日期', '预测销售单价(元/千克)']])
predictions[['单品类', '日期', '预测销售单价(元/千克)']].to_csv('predicted_prices_by_category.csv', index=False)
