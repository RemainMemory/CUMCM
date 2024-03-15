import pandas as pd
import matplotlib.pyplot as plt

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

# 创建中文到英文大类名称的映射字典
category_mapping = {
    '水生根茎类': 'Aquatic Roots and Stems',
    '花叶类': 'Flowers and Leaves',
    '花菜类': 'Cauliflower',
    '茄类': 'Eggplant',
    '辣椒类': 'Chili Peppers',
    '食用菌': 'Edible Fungi',
}

# 数据预处理：将数据转换为适用于折线图的格式
# 创建一个空的DataFrame，用于存储转换后的数据
monthly_sales = df.groupby(['月份', '大类'])['销量(千克)'].sum().unstack(fill_value=0)

# 将大类名称替换为英文
monthly_sales.columns = monthly_sales.columns.map(category_mapping)

# 将Period对象转换为字符串
monthly_sales.index = monthly_sales.index.strftime('%Y-%m')

# 绘制折线图
plt.figure(figsize=(12, 6))  # 设置图表大小
for category in monthly_sales.columns:
    plt.plot(monthly_sales.index, monthly_sales[category], label=category)

plt.title('Monthly Sales by Category')
plt.xlabel('Month')
plt.ylabel('Sales (Kilograms)')
plt.xticks(rotation=45)  # 旋转x轴标签以避免重叠
plt.legend(loc='upper right')  # 显示图例

# 显示图表
plt.tight_layout()
plt.show()
