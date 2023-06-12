import numpy as np


def linear_regression(X, y, method):  # X为nxp矩阵(n个样本,p个特征),且[1 X]列满秩
    n, p = X.shape
    X = np.hstack((np.ones((n, 1)), X))
    if method == 'inv':
        return np.linalg.inv(X.T @ X) @ X.T @ y  # 正规方程组直接求逆,也可以用LU分解
    elif method == 'qr':
        Q, R = np.linalg.qr(X, mode='complete')  # QR分解,这里R是nx(p+1)的
        return np.linalg.solve(R[:p + 1, :], (Q.T @ y)[:p + 1])  # p+1是因为X加了全一列，对应截距
    elif method == 'svd':
        U, S, VT = np.linalg.svd(X)  # Sigma是一维向量
        return np.linalg.solve(np.diag(S) @ VT, U[:, :p + 1].T @ y)


def lr_SGD(X, y, lr, gamma, n_epoches, batch_size):  # 带动量的小批量梯度下降，gamma=0时退化为小批量梯度下降
    n, p = X.shape
    X = np.hstack((np.ones((n, 1)), X))
    omega, delta = np.random.RandomState(43).randn(p + 1), 0  # delta为历史变化量的指数加权平均
    for epoch in range(n_epoches):
        for _ in range(n // batch_size):
            indexes = np.random.RandomState(43).permutation(n)[:batch_size]
            batch_X, batch_y = X[indexes], y[indexes]
            loss = ((batch_X @ omega - batch_y) ** 2).mean()
            grad = 2 / batch_size * batch_X.T @ (batch_X @ omega - batch_y)
            delta = gamma * delta - lr * grad  # gamma 为动量系数，利用当前点的梯度对上次迭代的变化量进行纠正
            omega += delta
        print(f'epoch {epoch}, loss {loss}')
    return omega


def lr_NAG(X, y, lr, gamma, n_epoches, batch_size):  # Nesterov’s Accelerated Gradient
    n, p = X.shape
    X = np.hstack((np.ones((n, 1)), X))
    omega, delta = np.random.RandomState(43).randn(p + 1), 0
    for epoch in range(n_epoches):
        for _ in range(n // batch_size):
            indexes = np.random.RandomState(43).permutation(n)[:batch_size]
            batch_X, batch_y = X[indexes], y[indexes]
            loss = ((batch_X @ omega - batch_y) ** 2).mean()
            omega_ahead = omega + gamma * delta  # 前瞻点，用作下个点的估计
            grad = 2 / batch_size * batch_X.T @ (batch_X @ omega_ahead - batch_y)
            delta = gamma * delta - lr * grad  # 利用前瞻点的梯度对上次迭代的变化量进行纠正
            omega += delta  # 注意在原点上更新，而不是前瞻点
        print(f'epoch {epoch}, loss {loss}')
    return omega


def lr_newton(X, y, T):  # 牛顿法
    n, p = X.shape
    X = np.hstack((np.ones((n, 1)), X))
    omega = np.random.RandomState(43).randn(p + 1)
    for t in range(T):
        loss = ((X @ omega - y) ** 2).mean()
        grad = 2 / n * X.T @ (X @ omega - y)
        H = 2 / n * X.T @ X  # Hessian矩阵
        omega -= np.linalg.inv(H) @ grad
        print(f't {t}, loss {loss}')
    return omega


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression

    np.random.seed(10)
    # y = 10 + 5 x1 + 3 x2
    n = 1000
    x_1 = np.linspace(-10, 10, n)
    x_2 = np.linspace(-10, 10, n)
    y = 10 + 5 * x_1 + 3 * x_2
    X = np.hstack((x_1[:, None] + np.random.normal(0, 2, (n, 1)), x_2[:, None] + np.random.normal(0, 1, (n, 1))))

    model = LinearRegression()
    model.fit(X, y)
    print(model.intercept_, model.coef_)

    omega = linear_regression(X, y, 'inv')
    print(omega)
    omega = linear_regression(X, y, 'qr')
    print(omega)
    omega = linear_regression(X, y, 'svd')
    print(omega)

    omega = lr_SGD(X, y, 1e-2, 0, 200, n)  # batch
    print(omega)

    omega = lr_SGD(X, y, 5e-5, 0, 10, 1)  # SGD
    print(omega)

    omega = lr_SGD(X, y, 5e-3, 0, 200, 64)  # mini-batch
    print(omega)

    omega = lr_SGD(X, y, 5e-3, 0.9, 10, 64)  # mini-batch with momentum
    print(omega)
    omega = lr_NAG(X, y, 5e-3, 0.9, 5, 64)  # mini-batch with Nesterov’s Accelerated Gradient
    print(omega)

    omega = lr_newton(X, y, 5)  # Newton method
    print(omega)
