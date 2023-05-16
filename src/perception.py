import numpy as np


def perception(X, y, lr=1e-3, max_iters=1000, seed=42):
    n, p = X.shape
    y[y == 0], iter = -1, 0
    w, b = np.random.RandomState(seed).random(p), np.random.RandomState(seed).random(1)
    while iter < max_iters:
        row, iter = np.random.randint(0, n), iter + 1
        if -y[row] * (w @ X[row, :] + b) > 0:  # SGD
            w += lr * y[row] * X[row, :]
            b += lr * y[row]
    return w, b


if __name__ == '__main__':
    np.random.seed(41)
    X = np.random.randint(-50, 50, (200, 4))
    w, b = np.array([3, 8, 21, -5]), np.array([-30])
    y = np.array([0 if row @ w + b > 0 else 1 for row in X])  # 构造线性可分数据集(X,y)

    w, b = perception(X, y.copy(), max_iters=2000)
    print(w, b)
    print(y)
    predict = (X @ w + b > 0).astype(int)
    print(predict)
    print((y == predict).sum() / 200)
