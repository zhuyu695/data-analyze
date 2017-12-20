import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from Util import plot_decision_regions

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# plt.scatter(X_xor[y_xor == 1, 0],
#             X_xor[y_xor == 1, 1],
#             c = 'b', marker = 'x', label = '1')
#
# plt.scatter(X_xor[y_xor == -1, 0],
#             X_xor[y_xor == -1, 1],
#             c = 'r', marker = 's', label = '-1')
#
# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.legend(loc = 'best')

svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()