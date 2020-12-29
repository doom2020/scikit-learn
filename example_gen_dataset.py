from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=1000, n_features=10, n_targets=1, noise=5)
print(X.shape)
print(y.shape)
lineReg = LinearRegression()
lineReg.fit(X, y)
print(lineReg.predict(X[:4,:]))
# 线性回归斜率
print(lineReg.coef_)
# 线性回归截距
print(lineReg.intercept_)