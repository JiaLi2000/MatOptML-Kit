import numpy as np


def TPR(y_pred: np.array, y: np.array):  # Recall
    return ((y_pred == 1) & (y == 1)).sum() / (y == 1).sum()


def FPR(y_pred: np.array, y: np.array):
    return ((y_pred == 1) & (y == 0)).sum() / (y == 0).sum()


def auc_roc(y_score: np.array, y: np.array):
    y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())  # 线性缩放至[0,1]
    pairs = []
    for eta in np.linspace(0, 1, num=10000, endpoint=True):
        y_pred = (y_score >= eta)
        tpr = TPR(y_pred, y)
        fpr = FPR(y_pred, y)
        pairs.append([fpr, tpr])
    area = 0
    pre_fpr, pre_tpr = pairs[0]
    for fpr, tpr in pairs[1:]:
        area += -(fpr - pre_fpr) * (pre_tpr + tpr) / 2  # fpr为eta的减函数
        pre_fpr, pre_tpr = fpr, tpr
    return area


def kmeans(X, k, T, eps, seed):  # kmeans++, O(Tnkd)
    np.random.seed(seed)
    n, d = X.shape
    centers_idxs = [np.random.randint(n)]  # 写上size将返回ndarray，不写则返回int
    for i in range(1, k):  # kmeans++ ，原始kmeans按np.random.randint(0, n, k)均匀采k个点
        min_dis_square = np.array(
            [[((x - X[center_idx, :]) ** 2).sum() for center_idx in centers_idxs] for x in X]).min(axis=1)
        p = min_dis_square / min_dis_square.sum()
        new_idx = np.random.choice(range(n), p=p)  # 按点到最近现存中心的距离平方和加权选择下一个中心点
        centers_idxs.append(new_idx)
    pre_centers, cur_centers = np.zeros((k, d)), X[np.array(centers_idxs), :]
    labels = np.zeros(n)
    for t in range(T):
        D = np.array([[np.linalg.norm(x - cur_center) for cur_center in cur_centers] for x in X])  # 点到中心的距离矩阵n x k
        labels = D.argmin(axis=1)  # 第一步,按就近原则为点分配簇所属
        pre_centers = cur_centers
        cur_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])  # 第二步，按簇所属更新簇中心
        if np.linalg.norm(cur_centers - pre_centers) < eps:
            break
    return cur_centers, labels


# def kmeans(X, k, T, eps):  # kmeans++, O(Tnkd)
#     n, d = X.shape
#     np.random.seed(43)
#     C_indexes = [np.random.randint(0, n)]  # kmeans++ 首个中心点从样本中随机选一个
#     for i in range(1, k):  # (原始kmeans按np.random.randint(0, n, k)均匀采k个点)
#         D_min = np.array([[((x - X[C_indexes[i], :]) ** 2).sum() for i in range(len(C_indexes))] for x in X]).min(
#             axis=1)
#         C_indexes.append(np.random.choice(n, 1, p=D_min / D_min.sum())[0])  # 按点到最近现存中心的距离平方和加权选择下一个中心点
#     C_new, C_old, label = X[C_indexes, :], X[C_indexes, :], np.zeros(n)
#     for t in range(T):
#         D = np.array([[((x - C_new[i]) ** 2).sum() for i in range(k)] for x in X])  # 各点到现有簇中心距离平方和矩阵
#         label = np.argmin(D, axis=1).flatten()  # 第一步,按就近原则为点分配簇所属
#         C_new, C_old = np.array([X[label == i].mean(axis=0) for i in range(k)]), C_new  # 第二步，按簇所属更新簇中心
#         if ((C_new - C_old) ** 2).sum() < eps:
#             break
#     return C_new, label


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    iris = load_iris()
    X = iris.data
    y = iris.target
    y[y == 2] = 0
    model = LogisticRegression()
    model.fit(X, y)
    y_score = model.predict_proba(X)[:, 1]
    print(roc_auc_score(y, y_score))
    print(auc_roc(y_score, y))

    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    iris = load_iris()
    X = iris.data
    model = KMeans(3)
    model.fit(X)
    print(model.cluster_centers_)
    print(model.labels_)
    centers, labels = kmeans(X, 3, 10000, 1e-9, 12)
    print(centers, labels)
