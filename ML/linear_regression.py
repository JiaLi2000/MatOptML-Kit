import numpy as np


def linear_regression(X, y, method):  # X为nxp矩阵(n个样本,p个特征),且[1 X]列满秩
    n, p = X.shape
    X = np.hstack((np.ones((n, 1)), X))
    if method == 'inv':
        return np.linalg.inv(X.T @ X) @ X.T @ y  # 正规方程组直接求逆,也可以用LU分解、Cholesky分解
    elif method == 'qr':
        Q, R = np.linalg.qr(X, mode='complete')  # QR分解,这里R是nx(p+1)的
        return np.linalg.solve(R[:p + 1, :], (Q.T @ y)[:p + 1])  # p+1是因为X加了全一列，对应截距
    elif method == 'svd':
        U, S, VT = np.linalg.svd(X)  # Sigma是一维向量
        return np.linalg.solve(np.diag(S) @ VT, U[:, :p + 1].T @ y)


def lr_Adam(X, y, lr, rho1, rho2, n_epoches, batch_size):  # Adam
    n, p = X.shape
    np.random.seed(43)
    X, eps, rho1k, rho2k = np.hstack((np.ones((n, 1)), X)), 1e-8, 1, 1
    omega, S, M = np.random.randn(p + 1), np.zeros(p + 1), np.zeros(p + 1)  #
    for epoch in range(n_epoches):
        for _ in range(n // batch_size):
            indexes = np.random.permutation(n)[:batch_size]
            batch_X, batch_y = X[indexes], y[indexes]
            loss = ((batch_X @ omega - batch_y) ** 2).mean()
            grad = 2 / batch_size * batch_X.T @ (batch_X @ omega - batch_y)
            S = rho1 * S + (1 - rho1) * grad  # Momentum
            M = rho2 * M + (1 - rho2) * grad ** 2  # RMSProp
            rho1k, rho2k = rho1k * rho1, rho2k * rho2
            S_hat, M_hat = S / (1 - rho1k), M / (1 - rho2k)  # 修正一阶矩，二阶矩
            omega -= lr / (np.sqrt(M_hat) + eps) * S_hat  # eps 防止下溢
        print(f'epoch {epoch}, loss {loss}')
    return omega


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression

    np.random.seed(12)
    # y = 10 + 5 x1 + 3 x2
    n = 10000
    x_1 = np.linspace(-10, 10, n)
    x_2 = np.linspace(-10, 10, n)
    y = 10 + 5 * x_1 + 3 * x_2
    X = np.hstack((x_1[:, None] + np.random.normal(0, 2, (n, 1)), x_2[:, None] + np.random.normal(0, 1, (n, 1))))

    model = LinearRegression()
    model.fit(X, y)
    print(model.intercept_, model.coef_)
    print(model.score(X, y))

    omega = linear_regression(X, y, 'inv')
    print(omega)
    import utils

    print(utils.r2(X @ omega[1:] + omega[0], y))

    omega = linear_regression(X, y, 'qr')
    print(omega)
    omega = linear_regression(X, y, 'svd')
    print(omega)

    omega = lr_Adam(X, y, 5e-1, 0.9, 0.999, 12, 64)  # Adam
    print(omega)
