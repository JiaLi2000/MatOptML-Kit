import numpy as np


# precision, recall, accurary, f1, auc, mse ,mae, kl,crossen,entropy

# regression

def mae(y_pre, y):
    return (np.abs(y - y_pre)).mean()


def mse(y_pre, y):
    return ((y - y_pre) ** 2).mean()


def rmse(y_pre, y):
    return ((y - y_pre) ** 2).mean() ** 0.5


def r2(y_pre, y):
    y_bar = y.mean()
    SSE = ((y - y_pre) ** 2).sum()
    SST = ((y - y_bar) ** 2).sum()
    return 1 - SSE / SST


# binary classification
def accuracy(y_pre, y):
    return np.count_nonzero(y_pre == y) / len(y)


def precision(y_pre, y):
    return np.count_nonzero((y == 1) & (y_pre == 1)) / np.count_nonzero(y_pre == 1)


def recall(y_pre, y):
    return np.count_nonzero((y == 1) & (y_pre == 1)) / np.count_nonzero(y == 1)


def f1(y_pre, y):
    p = precision(y_pre, y)
    r = recall(y_pre, y)
    return 2 * p * r / (p + r)


# others
def train_test_split(X, y, test_ratio, seed):  # X,y均为ndarray
    n = len(X)
    test_size = int(n * test_ratio)
    indexes = np.random.RandomState(seed).permutation(n)
    train_indexes = indexes[:test_size]
    test_indexes = indexes[test_size:]
    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]
