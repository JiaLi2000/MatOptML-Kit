import numpy as np


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


def recall(y_pre, y):  # TPR
    return np.count_nonzero((y == 1) & (y_pre == 1)) / np.count_nonzero(y == 1)


def FPR(y_pre, y):  # 假阳性预测个数/阴性总数
    return np.count_nonzero((y == 0) & (y_pre == 1)) / np.count_nonzero(y == 0)


def f1(y_pre, y):
    p = precision(y_pre, y)
    r = recall(y_pre, y)
    return 2 * p * r / (p + r)


def confusion_matrx(y_pre, y):  # C_ij表示真实为i而预测为j的个数
    C = np.zeros((2, 2))
    C[0, 0] = np.count_nonzero((y == 0) & (y_pre == 0))  # TN
    C[0, 1] = np.count_nonzero((y == 0) & (y_pre == 1))  # FP
    C[1, 0] = np.count_nonzero((y == 1) & (y_pre == 0))  # FN
    C[1, 1] = np.count_nonzero((y == 1) & (y_pre == 1))  # TP
    return C


def auc_roc(y_score, y):
    y_score_01 = (y_score - y.min()) / (y.max() - y.min())
    pairs = []  # (FPR,TPR/Recall) pairs
    for threshold in np.linspace(0, 1, 10000, endpoint=True):  # 根据阈值离散化,记录(FPR,TPR)对
        y_pre = (y_score_01 >= threshold).astype(int)
        TPR_ = recall(y_pre, y)
        FPR_ = FPR(y_pre, y)
        pairs.append([FPR_, TPR_])
    area = 0
    pre_FPR, pre_TPR = pairs[0]
    for cur_FPR, cur_TPR in pairs[1:]:  # FPR随阈值增大而减小
        area += (pre_TPR + cur_TPR) * -(cur_FPR - pre_FPR) / 2  # 梯形法则(小梯形面积求和)代替积分
        pre_FPR, pre_TPR = cur_FPR, cur_TPR  # 下一个小梯形
    return area


# others
def train_test_split(X, y, test_ratio, seed):  # X,y均为ndarray
    n = len(X)
    test_size = int(n * test_ratio)
    indexes = np.random.RandomState(seed).permutation(n)
    train_indexes = indexes[:test_size]
    test_indexes = indexes[test_size:]
    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]
