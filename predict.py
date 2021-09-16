"""
Created on Tue April 14 10:41:28 2021
@author: ZhangTeng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time
file1 = "errordata/Error.csv"
data = np.array(pd.read_csv(file1))
uv = data[:, 0:2]
n_source1 = 70

x_source1 = uv
x_source2 = uv
x_source3 = uv
x_source4 = uv
x_source5 = uv
x_source6 = uv
x_source7 = uv
x_source8 = uv

x_target9 = uv
x_target10 = uv
x_target11 = uv
x_target12 = uv

y_source1 = data[:, 2]
y_source2 = data[:, 3]
y_source3 = data[:, 4]
y_source4 = data[:, 5]
y_source5 = data[:, 6]
y_source6 = data[:, 7]
y_source7 = data[:, 8]
y_source8 = data[:, 9]

y_target9 = data[:, 10]
y_target10 = data[:, 11]
y_target11 = data[:, 12]
y_target12 = data[:, 13]

n_target_test = 70
x_target_test = x_target9
y_target_test = y_target9

n_target_train = 12
x_target_train = []
y_target_train = []
index = np.array([1, 5, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38])
for i in index:
    x_target_train.append(uv[i, :])
    y_target_train.append(y_target_test[i,])
x_target_train = np.array(x_target_train)
y_target_train = np.array(y_target_train)


X = np.concatenate(( x_source1,x_source2, x_source3,  x_source4, x_source5,  x_source6, x_source7, x_source8, x_target_train))
y = np.concatenate(( y_source1,y_source2, y_source3,  y_source4, y_source5,  y_source6, y_source7, y_source8, y_target_train))
sample_size = [ n_source1 + n_source1 +n_source1 + n_source1  + n_source1 + n_source1 + n_source1 + n_source1, n_target_train]

n_estimators =500
steps = 10
fold = 3
random_state = np.random.RandomState(10)

start = time.perf_counter()

regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6, random_state=144), n_estimators=n_estimators,
                              sample_size=sample_size, steps=steps, fold=fold, random_state=random_state)
regr_1.fit(X, y)

end = time.perf_counter()
print("训练用时：")
print('%.10f'%(end - start))

start = time.perf_counter()

y_pred1 = regr_1.predict(x_target_test)

end = time.perf_counter()
print("测试用时：")
print('%.10f'%(end - start))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(x_target_train[:, 0], x_target_train[:, 1], y_target_train, c='b', linewidths=3)
ax2.plot_trisurf(x_target_test[:, 0], x_target_test[:, 1], y_target_test, cmap='hot')
ax2.plot_trisurf(x_target_test[:, 0], x_target_test[:, 1], y_pred1, cmap='Greens')
ax2.set_xlabel("curve U")
ax2.set_ylabel("curve V")
ax2.set_zlabel("Error/μm")
plt.title("Target curve")
plt.show()

plt.figure()
plt.scatter(y_target_test, y_pred1, marker='o', edgecolors='g', s=50, c='#00CED1')
x = np.linspace(0, 150, 10)
y = x
y2 = 1.10 * x
y3 = 0.90 * x
plt.plot(x, y, color='red', linewidth=2.0, linestyle='--', label='line')
plt.plot(x, y2, color='blue', linewidth=1, linestyle='--', label='line')
plt.plot(x, y3, color='blue', linewidth=1, linestyle='--', label='line')
plt.legend(["y = x", 'Upper', 'Downer' ,"Predict_value"])
plt.xlabel('real value')
plt.ylabel('predict value')
plt.title('processiong error relative reference')
plt.show()

plt.figure()
plt.plot(y_target_test.ravel(), color='red', linewidth=1.0, linestyle='-', label='real value')
plt.plot(y_pred1.ravel(), color='blue', linewidth=1.0, linestyle='--', label='predict value')
plt.bar(x=range(len(y_pred1)), height=np.abs(y_target_test.ravel() - y_pred1.ravel()), color='blue', linewidth=1.0,
        linestyle=':', label='Abs Error')
plt.legend()
plt.title("Change rule")
plt.xlabel('num')
plt.ylabel('error/μm')
plt.show()

mse_twostageboost = mean_squared_error(y_target_test, y_pred1)
R2 = r2_score(y_target_test, y_pred1)
rmse = np.sqrt(mse_twostageboost)

print("MSE:", mse_twostageboost)
print("R2：", R2)
print("RMSE：", rmse)

