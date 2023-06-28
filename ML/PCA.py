import numpy as np


def pca(X, k):
    n, d = X.shape
    X -= X.mean(axis=0)
    Sigma = X.T @ X / (n - 1)  # 数据矩阵X列归一化后才表示协方差阵
    eigvalues, eigvectors = np.linalg.eigh(Sigma)
    W = eigvectors[:, range(-1, -k - 1, -1)]  # eigh默认升序排列，倒序取k个即topk特征向量
    # print(W) #主成分向量
    return X @ W  # 新坐标


if __name__ == '__main__':
    import sklearn.datasets as sd
    from sklearn.decomposition import PCA

    X = sd.load_iris().data[:, :3]
    model = PCA(2)
    model.fit(X)
    print(model.components_)
    new_X = pca(X, 2)
    print(new_X)
