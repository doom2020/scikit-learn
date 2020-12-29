from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt

# 导入波士顿房价数据
boston_house = datasets.load_boston()
# 获取所有样本以及属性
X_data = boston_house.data
# 获取对应样本的值
y_data = boston_house.target
# 使用线性回归算法
lineReg = LinearRegression()
# 训练模型
lineReg.fit(X_data, y_data)
# 预测模型
print(lineReg.predict(X_data[:4,:]))
# 和实际值对比
print(y_data[:4])
# 获取模型参数
params = lineReg.get_params()
# 获取模型分数
score = lineReg.score(X_data, y_data)
# 保存模型(2种方法)
import pickle
with open('boston_hourse.pickle', 'wb') as fw:
    fw.write(pickle.dumps(lineReg))

import joblib
joblib.dump(lineReg, 'boston_horse.joblib')



