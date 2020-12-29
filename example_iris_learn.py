from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import joblib

# 获取数据集里iris数据
iris = datasets.load_iris()
# 获取所有样本和特征: 二维数组样本数量以及样本4个属性值
iris_X = iris.data
# 获取对应样本属于的类别：3种
iris_y = iris.target

# 将数据集分为3、7比例分成测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

# 选择knn算法,创建knn实例
knn = KNeighborsClassifier()
# 训练模型
knn.fit(X_train, y_train)
# 预测值
# print(knn.predict(X_test))
# 对比一下实际值
# print(y_test)

# 保存模型2种方法(下次可直接使用)
with open('iris_model1.pick', 'wb') as fw:
    fw.write(pickle.dumps(knn))

joblib.dump(knn, 'iris_model2.joblib')

# 获取一下knn模型的一下属性(knn实例参数)
# 获取knn模型的参数
params = knn.get_params()
print(params)
# 获取knn模型的模型分数
score = knn.score(X_test, y_test)
print(score)

