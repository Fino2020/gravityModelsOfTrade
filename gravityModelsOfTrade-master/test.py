import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # 线性回归
import pandas as pd
# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
# data = [
#     [0.067732, 3.176513], [0.427810, 3.816464], [0.995731, 4.550095], [0.738336, 4.256571], [0.981083, 4.560815],
#     [0.526171, 3.929515], [0.378887, 3.526170], [0.033859, 3.156393], [0.132791, 3.110301], [0.138306, 3.149813],
#     [0.247809, 3.476346], [0.648270, 4.119688], [0.731209, 4.282233], [0.236833, 3.486582], [0.969788, 4.655492],
#     [0.607492, 3.965162], [0.358622, 3.514900], [0.147846, 3.125947], [0.637820, 4.094115], [0.230372, 3.476039],
#     [0.070237, 3.210610], [0.067154, 3.190612], [0.925577, 4.631504], [0.717733, 4.295890], [0.015371, 3.085028],
#     [0.335070, 3.448080], [0.040486, 3.167440], [0.212575, 3.364266], [0.617218, 3.993482], [0.541196, 3.891471]
# ]
#
# # 生成X和y矩阵
# dataMat = np.array(data)
# X = dataMat[:, 0:1]  # 变量x
# y = dataMat[:, 1]  # 变量y
# print(X, y)

csv = pd.read_csv('gravity.csv')
csv['log( (GDP_i * GDP_j)/distance)'].replace([np.inf, -np.inf], np.nan, inplace=True)
# print(csv[csv['log( (GDP_i * GDP_j)/distance)'].replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)])
csv.dropna(inplace=True)
print("csv:{}".format(csv['log( (GDP_i * GDP_j)/distance)']))
csv_list = csv.values
# print(csv_list)
# print(csv['log( (GDP_i * GDP_j)/distance)'])
# print(csv['log(exports)'])
csv_list = np.array(csv_list)

X = csv_list[:, -2:-1]
y = csv_list[:, -1]
print(X, y)

# # ========线性回归========
model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.fit(X, y)  # 线性回归建模
print('系数矩阵:\n', model.coef_)
print('线性回归模型:\n', model)
# 使用模型预测
predicted = model.predict(X)

plt.scatter(X, y, marker='x')
plt.plot(X, predicted, c='r')

plt.xlabel("x")
plt.ylabel("y")
plt.show()
