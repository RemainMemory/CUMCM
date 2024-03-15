import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 读取数据的10%样本
df = pd.read_csv('../sales_data0.csv').sample(frac=0.1, random_state=42)

# 将销售日期转换为Pandas的datetime格式
df['销售日期'] = pd.to_datetime(df['销售日期'])

# 按天和大类聚合数据
daily_transactions = df.groupby(['销售日期', '单品类'])['销量(千克)'].sum().reset_index()

# 将按天聚合的数据转换为适用于Apriori的格式
basket_by_day = daily_transactions.pivot_table(index='销售日期', columns='单品类', values='销量(千克)', aggfunc='sum',
                                               fill_value=0)

# 将销量转换为二进制购买指标
basket_sets_by_day = basket_by_day.apply(lambda x: x.map(lambda y: True if y > 0 else False))

# 使用Apriori算法找出频繁项集
frequent_itemsets_by_day = apriori(basket_sets_by_day, min_support=0.005, use_colnames=True)

# 检查是否找到了频繁项集
if frequent_itemsets_by_day.empty:
    print("No frequent itemsets found!")
else:
    # 生成关联规则
    rules_by_day = association_rules(frequent_itemsets_by_day, metric="lift", min_threshold=1)

    # 输出关联规则到Excel
    rules_by_day.to_csv('association_rules_by_day.csv', index=False)
