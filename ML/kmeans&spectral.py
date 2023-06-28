import numpy as np
import networkx as nx


def kmeans(X, k, T, eps):  # kmeans++, O(Tnkd)
    n, d = X.shape
    np.random.seed(43)
    C_indexes = [np.random.randint(0, n)]  # kmeans++ 首个中心点从样本中随机选一个
    for i in range(1, k):  # (原始kmeans按np.random.randint(0, n, k)均匀采k个点)
        D_min = np.array([[((x - C_indexes[i]) ** 2).sum() for i in range(len(C_indexes))] for x in X]).min(axis=1)
        C_indexes.append(np.random.choice(n, 1, p=D_min / D_min.sum())[0])  # 按点到最近现存中心的距离平方和加权选择下一个中心点
    C_new, C_old, label = X[C_indexes, :], X[C_indexes, :], np.zeros(n)
    for t in range(T):
        D = np.array([[((x - C_new[i]) ** 2).sum() for i in range(k)] for x in X])  # 各点到现有簇中心距离平方和矩阵
        label = np.argmin(D, axis=1).flatten()  # 第一步,按就近原则为点分配簇所属
        C_new, C_old = np.array([X[label == i].mean(axis=0) for i in range(k)]), C_new  # 第二步，按簇所属更新簇中心
        if ((C_new - C_old) ** 2).sum() < eps:
            break
    return C_new, label


def spectral(G, k, T, eps):  # min Ncut
    L = nx.laplacian_matrix(G, sorted(G.nodes)).toarray()  # L = D - W
    Dsqinv = np.diag(1 / L.diagonal() ** 2)  # 度矩阵的-1/2次幂
    L_sys = Dsqinv @ L @ Dsqinv
    eigvalues, eigvectors = np.linalg.eigh(L_sys)  # 对标准化的Laplacian特征分解
    H = Dsqinv @ eigvectors[:, :k]  # top-k small 的最优解 变换后 作为谱嵌入
    _, label = kmeans(H, k, T, eps)
    return label


if __name__ == "__main__":
    from sklearn.cluster import KMeans
    import sklearn.datasets

    X = sklearn.datasets.load_iris().data
    model = KMeans(n_clusters=3, n_init='auto')
    model.fit(X)
    print(model.cluster_centers_, model.labels_)
    C, labels = kmeans(X, 3, 1000, 1e-3)
    print(C, labels)

    import matplotlib.pyplot as plt

    G = nx.karate_club_graph()
    label = spectral(G, 3, 1000, 1e-3)
    print(label)
    pos = nx.spring_layout(G)  # 节点的布局为spring型
    plt.figure(figsize=(8, 6))  # 图片大小
    nx.draw_networkx(G, pos=pos, node_color=label)
    plt.show()
