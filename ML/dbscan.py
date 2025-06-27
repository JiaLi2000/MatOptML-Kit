import numpy as np


def dbscan(X, eps, m):  # 直接使用ndarray进行邻居查找的DBSCAN, O(n^2)
    n = X.shape[0]
    D = np.array([[((X[i, :] - X[j, :]) ** 2).sum() ** 0.5 for i in range(n)] for j in range(n)])  # nxn欧式距离矩阵
    label, cluster_id = np.full(n, -2), 0  # 用-2表示点处于未分配状态,-1表示异常，簇从0编号
    for i in range(n):  # 由定理1,从任一内部点密度可达的点构成一个簇,因此遍历一遍内部点即可确定所有簇
        if label[i] != -2:  # 已分配表示要么已分配的边界点、要么已分配的内部点。二者均不必再充当可能的“未分配内部点”遍历
            continue
        neighbors = np.where(D[i] <= eps)[0]  # 当前点i的eps邻域
        if len(neighbors) < m:  # 非内部点，可能是边界点也可能是异常点。暂时无法判断,统一设置为异常点(若是边界点后续会被重新修改)
            label[i] = -1
            continue
        label[i] = cluster_id  # 运行到这里,说明当前点i是内部点 且 标签未分配
        seeds = set(neighbors) - {i}  # 由定理1,从当前内部点密度可达(邻域内)的所有点同簇
        while seeds:  # 对邻域内除自身外的每个点,若没有标签,则赋予其及其邻域(内部点,二次传播)同样的簇标签
            j = seeds.pop()
            if label[j] not in [-1, -2]:  # 这里将前面暂时设置为异常的边界点修正
                continue
            neighbors = np.where(D[j] <= eps)[0]  # 传递性,若从点i密度可达的点j仍然是内部点，则进行二次传播
            if len(neighbors) >= m:
                seeds.update(set(neighbors) - {j})
            label[j] = cluster_id  # 在seeds中的元素是密度联通的
        cluster_id += 1
    return label


if __name__ == "__main__":
    from sklearn.cluster import DBSCAN
    import sklearn.datasets

    X = sklearn.datasets.load_iris().data
    model = DBSCAN()
    model.fit(X)
    print(model.labels_)
    label = dbscan(X, 0.5, 5)
    print(label)
