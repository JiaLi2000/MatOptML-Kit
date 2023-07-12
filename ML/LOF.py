import numpy as np


def lof(X, k):  # 计算局部异常值因子O(n^2),假设第k距离无重复
    n = X.shape[0]
    D = np.array([[((X[i, :] - X[j, :]) ** 2).sum() ** 0.5 for i in range(n)] for j in range(n)])  # nxn欧式距离矩阵
    D_sorted = D.argsort(axis=1)  # 对每个顶点,按所有点到它的距离升序排列
    kNN = D_sorted[:, 1:k + 1]  # 点的kNN不包括点本身,即排序后的第一个点
    kD = np.array([D[i, D_sorted[i, k]] for i in range(n)])  # k距离d_k(x)即点x到其第k近邻居的距离
    kRD = np.array([[max(kD[j], D[i, j]) for j in range(n)] for i in range(n)])  # k可达距离(x,o)=max{d(x,o),d_k(o)}
    LRD = np.array([1 / kRD[i, kNN[i, :]].mean() for i in range(n)])  # 局部可达密度=邻域平均可达距离倒数
    LOF = np.array([(LRD[kNN[i, :]] / LRD[i]).mean() for i in range(n)])  # LOF=平均邻域局部可达密度之比
    return LOF


if __name__ == "__main__":
    from sklearn.neighbors import LocalOutlierFactor
    import sklearn.datasets

    X = sklearn.datasets.load_iris().data
    model = LocalOutlierFactor(5)
    model.fit(X)
    print(-model.negative_outlier_factor_)
    lof(X, 5)
    scores = lof(X, 5)
    print(scores)
