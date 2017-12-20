import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data',
                 header=None)
df.tail()

import numpy as np

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-sentosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

# print(X)

import matplotlib.pyplot as plt

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='sentosa')
# plt.legend(loc='upper left')
# plt.show()

# 1. logistic regression
from LogisticRegression import AdalineGD

# fig, ax = plt.subplot(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)

# print(ada1.cost_)

# ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='0')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log(Sum-squared-error)')
# ax[0].set_title('Adaline - Learning rate 0.01')
#
# ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='0')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('log(Sum-squared-error)')
# ax[1].set_title('Adaline - Learning rate 0.01')
# plt.show()