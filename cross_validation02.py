from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np


digits = load_digits()
X = digits.data
y = digits.target

train_size, train_scores, test_scores = learning_curve(SVC(gamma=0.001), X, y, cv=10, scoring='neg_mean_squared_error', train_sizes=[0.1, 0.25, 0.5, 0.75])
train_loss_mean = -np.mean(train_scores, axis=1) # 取水平轴上的方差
test_loss_mean = -np.mean(test_scores, axis=1) # 取水平轴上的方差
print(train_scores)
print(test_scores)

plt.plot(train_size, train_loss_mean, 'o-', color="r", label='training')
plt.plot(train_size, test_loss_mean, 'o-', color='g', label='test')
plt.xlabel("training example")
plt.ylabel("loss")
plt.legend(loc='best')
plt.show()