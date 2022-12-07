import numpy as np
from nbc import DescretenaiveBayes 

# http://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29

def read_vine_data(filepath):
    D = np.genfromtxt(filepath, delimiter=",")
    y = D[:, ].astype(np.int8)
    X = D[:, 1:]
    return X, y 

def read_covertype_data(filepath):
    D = np.genfromtxt(filepath, delimiter=" ")
    y = D[:, 36].astype(np.int8)
    X = D[:, 1:35]
    return X, y 

def train_test_split(X, y, train_ratio=0.75, seed=0):
    np.random.seed(seed)
    m = X.shape[0]
    indexes = np.random.permutation(m)
    X = X[indexes]
    y = y[indexes]
    i = int(np.round(train_ratio * m))#indeks progowy
    X_train = X[:i]
    y_train = y[:i]

    X_test = X[i:]
    y_test = y[i:]

    return X_train, y_train, X_test, y_test

def discretize(X, bins=5, mins_ref=None, maxes_ref=None):
    if mins_ref is None:
        mins_ref = np.min(X, axis=0)
        maxes_ref = np.max(X, axis=0)
    X_d = np.clip((X - mins_ref) / (maxes_ref - mins_ref) * bins, 0, bins-1).astype(np.int8)
    return X_d, mins_ref, maxes_ref
    



data3 = "./wine.data"
data4 ="./sat.tst"
BINS = 5
Xwine, ywine = read_vine_data(data3)
X, y = read_covertype_data(data4)# read_vine_data(data3)
# Do testowania sytuacji niebezpiecznych
#X = np.tile(X, (1, 50))
X_train, y_train, X_test, y_test = train_test_split(X, y)
X_d_train, mins_ref, maxes_ref = discretize(X_train, BINS)
X_d_test, _, _ = discretize(X_test, bins=BINS, mins_ref=mins_ref, maxes_ref=maxes_ref)
n = X_train.shape[1]
domain_sizes = BINS * np.ones(n, dtype=np.int8)
clf = DescretenaiveBayes(domain_sizes, laplace=False, safe_count=False)
clf.fit(X_d_train, y_train)
print(clf.PY_)
predictions = clf.predict(X_d_test)
print(f"ACC TRAIN: {clf.score(X_d_train, y_train)}")
print(f"ACC Test: {clf.score(X_d_test, y_test)}")
print(predictions)
# X, y = read_vine_data(data)
# X_train, y_train, X_test, y_test = train_test_split(X, y)
# X_d_train, mins_ref, maxes_ref = discretize(X_train, BINS)
# X_d_test, _, _ = discretize(X_test, bins=BINS, mins_ref=mins_ref, maxes_ref=maxes_ref)
# n = X_train.shape[1]
# domain_sizes = BINS * np.ones(n, dtype=np.int8)
# clf = DescretenaiveBayes(domain_sizes)
# clf.fit(X_d_train, y_train)
# print(clf.PY_)
# predictions = clf.predict(X_d_test)
# print(f"ACC TRAIN: {clf.score(X_d_train, y_train)}")
# print(f"ACC Test: {clf.score(X_d_test, y_test)}")
# print(predictions)
#print(ywine)
#print(clf.scores)
# print(X_d_test)
# print(X_d_train)
# print(y_train)