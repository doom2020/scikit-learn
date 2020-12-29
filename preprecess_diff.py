from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.datasets.samples_generator import make_classification
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt



# 生成具有4种属性的1000笔分类数据
X, y = make_classification(n_samples=300, n_features=4, scale=10)
# 缩放预处理(方法一)
X = preprocessing.StandardScaler().fit(X).transform(X)
# 缩放预处理(方法二)两者结果相同
# X = preprocessing.scale(X)
print(X.shape)
print(y.shape)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

svc = SVC()
svc.fit(X_train, y_train)
print(svc.predict(X_test))
print(y_test)
print(svc.score(X_test, y_test))