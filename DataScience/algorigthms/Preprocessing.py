import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

# 1. Mapping ordinal feature:
size_mapping = { 'XL': 3, 'L': 2, 'M': 1 }
df['size'] = df['size'].map(size_mapping)
# print(df)

# 2. Mapping class labels:
class_mapping = { label: idx for idx, label in enumerate(np.unique(df['classlabel'])) }
df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)
inv_class_mapping = { v: k for k, v in class_mapping.items() }
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# print(df)

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)

class_le.inverse_transform(y)

# 3. One hot encoding:
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()[:, 1:]

pd.get_dummies(df[['color', 'size', 'price']], drop_first=True)

# 4. split
df = pd.read_csv('/Users/leon_zhu/PycharmProjects/data/wine.data', header=None)
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
              'Proline']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# 5. feature scaling
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# 6. SBS
df = pd.read_csv('/Users/leon_zhu/PycharmProjects/data/wine.data', header=None)
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
              'Proline']

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  StandardScaler
from SBS import SBS

X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.7, 1.02])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.show()
k3 = list(sbs.subsets_[10])
print(k3)

knn.fit(X_train_std, y_train)
# print('Training accuracy:', knn.score(X_train_std, y_train))
# print('Test accuracy:', knn.score(X_test_std, y_test))
#
# knn.fit(X_train_std[:, k3], y_train)
# print('Training accuracy:', knn.score(X_train_std[: k3], y_train))
# print('Test accuracy:', knn.score(X_test_std[: k3], y_test))

# 7. Feature importance with Random Forests
from sklearn.ensemble import RandomForestClassifier

feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importance = forest.feature_importances_
indices = np.argsort(importance)[::-1]
# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))
# plt.title('Feature Importance')
# plt.bar(range(X_train.shape[1]), importance[indices], align='center')
# plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# plt.show()

# 8. Select Features from model
from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
# print('Number of samples that meet this criterion:', X_selected.shape[0])
# for f in range(X_selected.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))
