import pandas as pd
from deap import base, creator, tools, algorithms
import random

df = pd.read_csv('predicted_prices_by_category1_adjust.csv')  # 请替换为您的CSV文件名

# 按照单品类进行分组，并对预测销售单价和销量进行求和
df = df.groupby('单品类').agg({'预测销售单价(元/千克)': 'sum', '销量': 'sum'}).reset_index()

# 创建问题类型 - 最大化问题
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 初始化单品的补货量
toolbox.register("attr_float", random.uniform, 2.5, 50)  # 补货量在2.5-50千克

# 初始化个体和种群
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=len(df))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 适应度函数
def evaluate(individual):
    total_sales = 0
    for i, qty in enumerate(individual):
        if qty <= 0:  # 确保进货量大于0
            return -1,  # 返回一个非常低的适应度值
        unit_price = df.loc[i, '预测销售单价(元/千克)']
        total_sales += unit_price * qty
    return total_sales,

# 交叉、变异和选择操作
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

if __name__ == "__main__":
    # 创建初始种群
    pop = toolbox.population(n=50)

    # 运行遗传算法
    algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.1, ngen=500,
                        stats=None, halloffame=None, verbose=True)

    # 输出最佳解
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual for 7 days is:", best_ind.fitness.values)  # 输出3：7天的最佳个体和其适应度值（总销售额）

    # 计算7月1日一天的最优补货量
    best_ind_one_day = [x / 7 for x in best_ind]

    # 详细解释最佳解
    print("Detailed explanation of the best solution for one day (July 1st):")
    for i in range(len(best_ind_one_day)):
        print(
            f"  - For the product {df.loc[i, '单品类']}, the optimal replenishment quantity "
            f"for July 1st is {best_ind_one_day[i]:.2f} kg.")  # 输出4：7月1日每个单品的最优补货量

# 将最佳解保存为CSV文件
best_solution_df = pd.DataFrame({
    'Product_Category': df['单品类'],
    'Optimal_Replenishment_Quantity_for_July_1': best_ind_one_day
})

best_solution_df.to_csv('optimal_replenishment_solution.csv', index=False)
