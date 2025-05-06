import numpy as np
from scipy.stats import rv_discrete

# 假设我们有两个离散分布，分布A和分布B
# 定义分布A的概率质量函数PMF
values_a = [1, 2, 3]  # 分布A的取值
probs_a = [0.2, 0.5, 0.3]  # 对应的概率

# 定义分布B的概率质量函数PMF
values_b = [1, 2, 3, 4]  # 分布B的取值
probs_b = [0.1, 0.2, 0.3, 0.4]  # 对应的概率

# 创建分布对象
dist_a = rv_discrete(values=(values_a, probs_a), name='DistributionA')
dist_b = rv_discrete(values=(values_b, probs_b), name='DistributionB')

# 计算联合分布的PMF
# 由于这里我们只是展示如何定义，实际上计算联合分布需要知道两个分布之间的关系
# 如果两个分布是独立的，联合PMF是各自PMF的乘积
# 这里我们只是展示如何计算一个分布的PMF
joint_pmf = np.outer(dist_a.pmf(values_a), dist_b.pmf(values_b))

# 打印联合分布的PMF
print("联合分布的PMF:")
for i, value_a in enumerate(values_a):
    for j, value_b in enumerate(values_b):
        print(f"P({value_a}, {value_b}) = {joint_pmf[i, j]}")