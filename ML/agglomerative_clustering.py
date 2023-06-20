import copy

import numpy as np


def agglomerative_clustering(X):
    n, d = X.shape
    dmin = lambda A, B: np.array([[((X[a, :] - X[b, :]) ** 2).sum() ** 0.5 for b in B] for a in A]).min()  # 最短距离
    tree = [[[i] for i in range(n)]]  # 1.初始时各点自成一簇
    while len(tree[-1]) != 1:
        D = np.array([[dmin(A, B) for B in tree[-1]] for A in tree[-1]])  # 2.簇间距离矩阵
        np.fill_diagonal(D, 1e8)  # 对角线为0，但不能选自身
        c1, c2 = sorted(np.unravel_index(np.argmin(D, axis=None), D.shape))  # 3.最小元对应的两个簇
        temp = copy.deepcopy(tree[-1])  # 在上一级的划分基础上构建新划分
        C1 = temp.pop(c1)
        C2 = temp.pop(c2 - 1)  # pop 小下标，则原大下标对应元素下标左移1位
        temp.append(C1 + C2)  # 合并两个簇
        tree.append(temp)
    return tree  # 输出层次树


if __name__ == "__main__":
    from sklearn.cluster import AgglomerativeClustering
    import sklearn.datasets

    X = sklearn.datasets.load_iris().data
    model = AgglomerativeClustering(4, linkage='single')
    model.fit(X)

    print(model.labels_)
    print(model.labels_[106], model.labels_[117], model.labels_[131])
    tree = agglomerative_clustering(X)
    print(tree[-4])
