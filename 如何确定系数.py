import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


data = pd.read_csv(r"train_data.csv")


# 假设您的数据列名为 'x' 和 'y'
x = data['x'].values
y = data['y'].values

# 第一步：拟合线性部分 y = kx
def linear_func(x, k, D):
    return k * x + D


# 拟合线性部分
k_opt, _ = curve_fit(linear_func, x, y)
y_linear_fit = linear_func(x, *k_opt)


print("线性拟合：")
# 绘制结果
plt.scatter(x, y, label='Data')
plt.plot(x, y_linear_fit, color='red', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# 第二步：拟合残差部分的周期性波动 y = A*sin(Bx + C)
def sin_func(x, A, B, C):
    return A * np.sin(B * x + C)


# 拟合周期性波动部分
initial_guess_sin = [10, 0.1, 0]  # 对 A, B, C 的初始猜测
params_sin, _ = curve_fit(sin_func, x, y, p0=initial_guess_sin)
y_sin_fit = sin_func(x, *params_sin)

print("周期拟合：")
# 绘制结果
plt.scatter(x, y_sin_fit, label='Data')
plt.plot(x, y_sin_fit, color='red', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 最终拟合结果
y_final_fit = y_linear_fit + y_sin_fit

# 绘制结果
plt.scatter(x, y, label='Data')
plt.plot(x, y_final_fit, color='red', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 输出拟合参数
print(f"Fitted linear slope: k = {k_opt[0]}, D = {k_opt[1]}")
print(f"Fitted sinusoidal parameters: A = {params_sin[0]}, B = {params_sin[1]}, C = {params_sin[2]}")

# 计算拟合损失
loss = mean_squared_error(y, y_final_fit)
print(loss)



# 生成10个70到100之间的随机数进行预测
x2 = np.sort(np.random.uniform(70, 100, 50))
print(x2)

y_pred_l = linear_func(x2, k_opt)
y_pred_s = sin_func(x2, *params_sin)
y_pred = y_pred_l + y_pred_s

# 绘制结果
plt.scatter(x, y, label='Data')
plt.plot(x, y_final_fit, color='red', label='Fitted curve')
plt.plot(x2, y_pred, color='blue', label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()