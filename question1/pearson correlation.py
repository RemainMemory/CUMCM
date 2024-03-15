import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据（假设您的数据保存在一个名为"data.csv"的CSV文件中）
df = pd.read_csv("../sales_data0.csv")

# 将销售日期转换为pandas的datetime对象，以便进行时间序列分析
df['销售日期'] = pd.to_datetime(df['销售日期'])

# 筛选出从2020年7月到2023年6月的数据
start_date = pd.to_datetime("2020-07-01")
end_date = pd.to_datetime("2023-06-30")
df = df[(df['销售日期'] >= start_date) & (df['销售日期'] <= end_date)]

# 提取月份信息
df['月份'] = df['销售日期'].dt.to_period('M')

# 数据预处理：将数据转换为适用于皮尔逊相关系数分析的格式
# 创建一个空的DataFrame，用于存储转换后的数据
basket = pd.DataFrame()

# 将每一笔交易转换为一个项集，这次按照“月份”和“大类”进行分组
for month, group in df.groupby('月份'):
    for category, sub_group in group.groupby('大类'):
        basket.at[month, category] = sub_group['销量(千克)'].sum()

# 将NaN值替换为0
basket.fillna(0, inplace=True)

# 计算皮尔逊相关系数
correlation_matrix = basket.corr(method='pearson')

# 将中文大类名称替换为英文名称的字典
category_mapping = {
    "水生根茎类": "ARR",
    "花叶类": "FL",
    "花菜类": "C",
    "茄类": "E",
    "辣椒类": "CP",
    "食用菌": "EM"
}

# 使用字典将大类名称替换为英文名称
correlation_matrix.rename(index=category_mapping, columns=category_mapping, inplace=True)

# 创建热力图
plt.figure(figsize=(10, 8))  # 设置图表大小
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# 将横坐标上的文字横着显示
plt.xticks(rotation=0)  # 将横坐标上的文字旋转为水平显示

plt.title('Pearson Correlation Coefficients')
plt.xlabel('Categories')
plt.ylabel('Categories')

# 显示图表
plt.show()

# 创建DataFrame以存储相关系数矩阵
correlation_df = pd.DataFrame(correlation_matrix, columns=basket.columns, index=basket.columns)

# 指定要保存的Excel文件名
output_file = "correlation_matrix.xlsx"

# 将相关系数矩阵保存到Excel文件
correlation_df.to_excel(output_file)

print(f"相关系数矩阵已保存到 {output_file}")