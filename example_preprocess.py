from sklearn.preprocessing import StandardScaler, scale
import numpy as np

# 缩放预处理数据(样本对应的属性相差特别大),还有其他很多预处理方式,根据实际情况选择
a = np.array([[10, 2.7, 3.6], [-100, 5, -2], [120, 20, 40]])

# 方法一
print(scale(a))
# 方法二
scaler = StandardScaler().fit(a)
print(scaler.transform(a))
