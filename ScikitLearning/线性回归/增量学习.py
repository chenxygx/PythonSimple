import numpy as np
from sklearn.linear_model import SGDRegressor
from random import randint

reg = SGDRegressor(loss="epsilon_insensitive", penalty="none", tol=1e-15)
X = np.linspace(0, 1, 50)
y = X / 2 + 0.3 + np.random.normal(0, 0.15, len(X))
X = X.reshape(-1, 1)

for i in range(10000):
    idx = randint(0, len(y) - 1)
    reg.partial_fit(X[idx:idx + 10], y[idx:idx + 10])

print(reg.coef_)
print(reg.intercept_)
