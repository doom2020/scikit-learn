from sklearn.datasets import load_iris # iris数据集
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.neighbors import KNeighborsClassifier # K最近邻(kNN，k-NearestNeighbor)分类算法

#加载iris数据集
iris = load_iris()
X = iris.data
y = iris.target

#分割数据并
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

#建立模型
# knn = KNeighborsClassifier()

# #训练模型
# knn.fit(X_train, y_train)

# #将准确率打印出
# # print(knn.score(X_test, y_test))
# # 0.973684210526
# # 导入交叉验证模块
# from sklearn.model_selection import cross_val_score
# # scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
import matplotlib.pyplot as plt #可视化模块

#建立测试参数集
k_range = range(1, 31)

k_scores = []

#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
from sklearn.model_selection import cross_val_score
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

#可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
# 一般来说准确率(accuracy)会用于判断分类(Classification)模型的好坏

