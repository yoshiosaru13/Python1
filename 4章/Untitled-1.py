import numpy as np
import matplotlib.pyplot as plt

# L2正則化による重みの制約を示すグラフの例
# x軸に重みパラメータの値、y軸に目的関数の値をプロット
# L2正則化を適用すると、重みパラメータの値が小さくなることを示す
weights = np.linspace(-5, 5, 100)  # -5から5までの重みパラメータの値を100等間隔に生成
objective_values = 0.5 * weights**2  # 目的関数の値を計算（二次関数の場合）
l2_reg = 0.5 * 0.1 * weights**2  # L2正則化の項を計算（正則化強度を0.1と仮定）
total_objective_values = objective_values + l2_reg  # 目的関数とL2正則化項を合計

plt.plot(weights, objective_values, label='Objective Function')
plt.plot(weights, l2_reg, label='L2 Regularization')
plt.plot(weights, total_objective_values, label='Total Objective Function')
plt.xlabel('Weights')
plt.ylabel('Objective Function')
plt.title('L2 Regularization')
plt.legend()
plt.show()




