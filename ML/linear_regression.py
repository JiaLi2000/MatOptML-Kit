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
        U, Sigma, VT = np.linalg.svd(X)  # Sigma是一维向量
        return np.linalg.solve(np.diag(Sigma) @ VT, U[:, :p + 1].T @ y)


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression

    np.random.seed(10)
    # y = 10 + 5 x1 + 3 x2
    n = 1000
    x_1 = np.linspace(-10, 10, n)
    x_2 = np.linspace(-10, 10, n)
    y = 10 + 5 * x_1 + 3 * x_2
    X = np.hstack((x_1[:, None] + np.random.normal(0, 2, (n, 1)), x_2[:, None] + np.random.normal(0, 1, (n, 1))))

    n, p = X.shape

    model = LinearRegression()
    model.fit(X, y)
    print(model.intercept_, model.coef_)

    omega = linear_regression(X, y, 'inv')
    print(omega)
    omega = linear_regression(X, y, 'qr')
    print(omega)
    omega = linear_regression(X, y, 'svd')
    print(omega)
