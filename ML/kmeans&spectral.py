import numpy as np
import networkx as nx

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
        labels = D.argmin(axis=1) # 第一步,按就近原则为点分配簇所属
        pre_centers = cur_centers
        cur_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)]) # 第二步，按簇所属更新簇中心
        if np.linalg.norm(cur_centers - pre_centers) < eps:
            break
    return cur_centers, labels

def spectral(G, k, T, eps):  # min Ncut
    L = nx.laplacian_matrix(G, sorted(G.nodes)).toarray()  # L = D - W
    Dsqinv = np.diag(1 / L.diagonal() ** 2)  # 度矩阵的-1/2次幂
    L_sys = Dsqinv @ L @ Dsqinv
    eigvalues, eigvectors = np.linalg.eigh(L_sys)  # 对标准化的Laplacian特征分解
    H = Dsqinv @ eigvectors[:, :k]  # top-k small 的最优解 变换后 作为谱嵌入
    _, label = kmeans(H, k, T, eps, 43)
    return label


if __name__ == "__main__":
    from sklearn.cluster import KMeans
    import sklearn.datasets

    X = sklearn.datasets.load_iris().data
    model = KMeans(n_clusters=3)
    model.fit(X)
    print(model.cluster_centers_, model.labels_)

    centers, labels = kmeans(X, 3, 10000, 1e-9, 12)
    print(centers, labels)
    import matplotlib.pyplot as plt

    G = nx.karate_club_graph()
    label = spectral(G, 3, 1000, 1e-3)
    print(label)
    pos = nx.spring_layout(G)  # 节点的布局为spring型
    plt.figure(figsize=(8, 6))  # 图片大小
    nx.draw_networkx(G, pos=pos, node_color=label)
    plt.show()
