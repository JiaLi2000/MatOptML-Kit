import numpy as np
import networkx as nx
import scipy.sparse as sp


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


def FPR(y_pre, y):  # 假阳性预测个数/阴性总数，误报率
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
    for cur_FPR, cur_TPR in pairs[1:]:  # recall随阈值增加而减小，阈值0对应(1,1), 阈值1对应(0,0)
        area += (pre_TPR + cur_TPR) * -(cur_FPR - pre_FPR) / 2  # 梯形法则(小梯形面积求和)代替积分
        pre_FPR, pre_TPR = cur_FPR, cur_TPR  # 下一个小梯形
    return area


# anomaly detection
def precision_at_k(y_pre, y, k):  # top-k异常值分数中真正为异常的占比
    topk_indexs = np.argsort(y_pre)[::-1][:k]  # top-k异常值分数的下标
    return np.count_nonzero(y[topk_indexs] == 1) / k


def recall_at_k(y_pre, y, k):  # 预测top-k中真异常值数与总真异常值数之比
    n_anomalies = np.count_nonzero(y)
    topk_indexs = np.argsort(y_pre)[::-1][:k]
    return np.count_nonzero((y[topk_indexs] == 1) & (y_pre[topk_indexs] == 1)) / n_anomalies


# clustering
def entropy(y):  # 聚类指示向量对应k点分布的熵
    _, cluster_counts = np.unique(y, return_counts=True)
    p = cluster_counts / cluster_counts.sum()
    return (-p * np.log2(p)).sum()


def joint_entropy(y_pre, y):  # 两聚类指派向量对应分布的联合熵
    joint = np.array(list(zip(y_pre, y)))
    _, cluster_counts = np.unique(joint, axis=0, return_counts=True)
    p = cluster_counts / cluster_counts.sum()
    return (-p * np.log2(p)).sum()


def NMI(y_pre, y):  # 归一化互信息,要求有聚类标签(忽略排列)
    entropy_y_pre = entropy(y_pre)
    entropy_y = entropy(y)
    joint_entropy_ = joint_entropy(y_pre, y)  # MI(X,Y) = 熵(X) + 熵(Y) - 联合熵(X,Y)
    return 2 * (entropy_y_pre + entropy_y - joint_entropy_) / (entropy_y_pre + entropy_y)


def silhouette(X, labels):  # 轮廓系数
    k = len(np.unique(labels))
    a_self = lambda x, X_: (((x - X_) ** 2).sum(axis=1) ** 0.5).sum() / (len(X_) - 1)  # 点x到X_中点平均距离(X包含x,结果不含x)
    a_other = lambda x, X_: (((x - X_) ** 2).sum(axis=1) ** 0.5).sum() / len(X_)  # 点x到X_中点平均距离(X不含x)
    b = lambda i: np.array(
        [a_other(X[i, :], X[labels == j]) for j in range(k) if j != labels[i]]).min()  # 点i到其他簇的最小平均距离
    s = lambda i: (b(i) - a_self(X[i, :], X[labels == labels[i]])) / max(b(i),
                                                                         a_self(X[i, :], X[labels == labels[i]]))
    return np.array([s(i) for i in range(len(X))]).mean()


def ncut(G: nx.graph, phi):
    n, W = len(phi), nx.adjacency_matrix(G, sorted(G.nodes))
    k = phi.max() + 1
    Y_0 = sp.coo_matrix(([1] * n, (range(n), phi)), dtype=int).tocsr()
    D = W.dot(Y_0).toarray()  # d_ij = sum_v of W_{iv}, where v is in cluster j
    degrees = D.sum(axis=1)
    volumes = np.array([degrees[phi == j].sum() for j in range(k)])
    cuts = volumes - (D * Y_0.toarray()).sum(axis=0)  # the right item just is association of clusters.
    f = (cuts / volumes).sum()
    return f


def modularity(G: nx.Graph, label):  # 无向带权图模块度计算
    n, W = len(label), nx.adjacency_matrix(G, sorted(G.nodes))
    k = label.max() + 1
    Y = sp.coo_matrix(([1] * n, (range(n), label)), dtype=int).tocsr()
    D = W.dot(Y).toarray()  # d_ij = sum_v of W_{iv}, where v is in cluster j
    degrees = D.sum(axis=1)
    m = degrees.sum() / 2
    volumes = np.array([degrees[label == j].sum() for j in range(k)])
    asso = (D * Y.toarray()).sum(axis=0)  # the right item just is association of clusters.
    Q = (asso / (2 * m) - (volumes / (2 * m)) ** 2).sum()
    return Q


# others
def train_test_split(X, y, test_ratio, seed):  # X,y均为ndarray
    n = len(X)
    test_size = int(n * test_ratio)
    indexes = np.random.RandomState(seed).permutation(n)
    train_indexes = indexes[:test_size]
    test_indexes = indexes[test_size:]
    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]
