import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
scx = StandardScaler();
scy = StandardScaler();
x = scx.fit_transform(x);
y = scy.fit_transform(y);

print(x)
print('___________________________________________________________________________________________________________________________')
print(y)

from sklearn.svm import SVR
reg = SVR(kernel='rbf')
reg.fit(x,y)

scy.inverse_transform(reg.predict(scx.transform([[6.5]])).reshape(-1,1))

plt.scatter(scx.inverse_transform(x), scy.inverse_transform(y), color='red')
plt.plot(scx.inverse_transform(x), scy.inverse_transform(reg.predict(x).reshape(-1,1)), color='blue')
plt.title('SVRegression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

x_grid = np.arange(min(scx.inverse_transform(x)), max(scx.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid)), 1)
plt.scatter(scx.inverse_transform(x), scy.inverse_transform(y), color='red')
plt.plot(x_grid, scy.inverse_transform(reg.predict(scx.transform(x_grid)).reshape(-1, 1)), color='blue')
plt.title('SVRegression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()