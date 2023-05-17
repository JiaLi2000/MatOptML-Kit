import numpy as np


def knn_classificaiton(X, y, query_vectors, k):
    results = []
    for q in query_vectors:
        dists = np.array([((x - q) ** 2).sum() ** 0.5 for x in X])
        knn_labels = y[np.argsort(dists)[:k]]
        results.append(np.bincount(knn_labels).argmax())  # 分类决策规则： 投票
    return results


def knn_regression(X, y, query_vectors, k):
    results = []
    for q in query_vectors:
        dists = np.array([((x - q) ** 2).sum() ** 0.5 for x in X])
        knn_labels = y[np.argsort(dists)[:k]]
        results.append(knn_labels.mean())  # 回归决策规则： 平均
    return results


if __name__ == '__main__':
    # 分类
    np.random.seed(41)
    X = np.random.randint(-50, 50, (200, 4))
    w, b = np.array([3, 8, 21, -5]), np.array([-30])
    y = np.array([0 if row @ w + b > 0 else 1 for row in X])

    query_vectors = X
    results = knn_classificaiton(X, y, query_vectors, 5)
    print(y)
    print(results)
    print((y == results).sum() / len(results))

    # 回归
    print('--------------')
    np.random.seed(41)
    X = np.random.randint(-50, 50, (200, 4))
    w, b = np.array([3, 8, 21, -5]), np.array([-30])
    y = X @ w + b + np.random.normal(0, 20, 200)

    query_vectors = X
    results = knn_regression(X, y, query_vectors, 2)
    print(list(y))
    print(results)

    print(((np.array(results) - y) ** 2).mean())
