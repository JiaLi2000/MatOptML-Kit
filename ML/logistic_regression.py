import numpy as np


def logistic_regression_SGD(X, y, lr, gamma, n_epoches, batch_size):  # 带动量的小批量梯度下降，gamma=0时退化为小批量梯度下降
    n, p = X.shape
    np.random.seed(43)
    X = np.hstack((np.ones((n, 1)), X))
    omega, delta = np.random.randn(p + 1), 0  # delta为历史变化量的指数加权平均
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    for epoch in range(n_epoches):
        for _ in range(n // batch_size):
            indexes = np.random.permutation(n)[:batch_size]
            batch_X, batch_y = X[indexes], y[indexes]
            loss = (-batch_y @ batch_X @ omega + np.logaddexp(0, batch_X @ omega).sum()) / batch_size
            grad = 1 / batch_size * batch_X.T @ (sigmoid(batch_X @ omega) - batch_y)
            delta = gamma * delta - lr * grad  # gamma 为动量系数，利用当前点的梯度对上次迭代的变化量进行纠正
            omega += delta
        print(f'epoch {epoch}, loss {loss}')
    return omega


def logistic_regression_newton(X, y, T):  # 牛顿法
    n, p = X.shape
    X = np.hstack((np.ones((n, 1)), X))
    omega = np.random.RandomState(43).randn(p + 1)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
    for t in range(T):
        loss = (-y @ X @ omega + np.logaddexp(0, X @ omega).sum()) / n
        grad = 1 / n * X.T @ (sigmoid(X @ omega) - y)
        H = 1 / n * X.T @ np.diag(sigmoid_prime(X @ omega)) @ X  # Hessian矩阵
        omega -= np.linalg.inv(H) @ grad
        print(f't {t}, loss {loss}')
    return omega


if __name__ == '__main__':
    np.random.seed(41)
    from sklearn.linear_model import LogisticRegression
    import sklearn.datasets as sd

    # iris = sd.load_iris()
    # X = iris.data
    # y = iris.target
    # y[y == 2] = 1
    import numpy as np
    import matplotlib.pyplot as plt

    # 生成随机数据
    np.random.seed(43)
    n = 1000
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # 绘制散点图
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    model = LogisticRegression(penalty=None,solver='sag')
    model.fit(X, y)
    print(model.intercept_, model.coef_)

    model = LogisticRegression(penalty=None)
    model.fit(X, y)
    print(model.intercept_, model.coef_)

    omega = logistic_regression_SGD(X, y, 1e-3, 0, 1000, len(y))  # mini-batch with momentum
    print(omega)

    omega = logistic_regression_newton(X, y, 10)  # Newton method
    print(omega)
