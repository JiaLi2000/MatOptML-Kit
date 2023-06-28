import numpy as np


def LASSO(X, y, T, mu):  # min 1/n ||Xw + b -y||_2^2 + \mu ||w||_1 坐标下降法CD求LASSO问题
    n, p = X.shape
    X = np.hstack((np.ones((n, 1)), X))  # 为X添加全1列将截距项合并入w
    np.random.seed(43)
    omega = np.random.randn(p + 1)
    for t in range(T):
        for i in range(p + 1):
            mi = 2 / n * (X[:, i] ** 2).sum()  # g_i'(x)中x的系数
            ci = X[:, np.arange(p + 1) != i] @ omega[np.arange(p + 1) != i] - y  # 用omega更新后的新值用来求解下一个子问题
            pi = 2 / n * X[:, i] @ ci  # g_i'(x)中x的常数
            if i == 0:
                omega[i] = -pi / mi  # 不对截距项惩罚,对应截距项凸可微子问题不含次梯度
            else:  # 利用凸不可微子问题最优性一阶充要条件，通过分类讨论求得最优解
                if pi > mu:
                    omega[i] = -(pi - mu) / mi
                elif pi < -mu:
                    omega[i] = -(pi + mu) / mi
                else:
                    omega[i] = 0
        loss = ((X @ omega - y) ** 2).mean() + mu * (np.abs(omega)).sum()
        print(f'iter {t} loss {loss}')
    return omega


def ridge_regression(X, y, mu):  # min  ||Xw + b -y||_2^2 + \mu ||w||^2_2 精确解
    n, p = X.shape
    X, I = np.hstack((np.ones((n, 1)), X)), np.eye(p + 1)
    I[0] = 0  # 不对截距项惩罚,充要条件中关于截距的梯度不变
    return np.linalg.inv(X.T @ X + mu * I) @ X.T @ y  # 正规方程组直接求逆,也可以用LU分解、Cholesky分解


def ridge_regression_Adam(X, y, mu, lr, rho1, rho2, n_epoches,
                          batch_size):  # min 1/n ||Xw + b -y||_2^2 + \mu ||w||^2_2
    n, p = X.shape
    np.random.seed(43)
    X, eps, rho1k, rho2k = np.hstack((np.ones((n, 1)), X)), 1e-8, 1, 1
    omega, S, M = np.random.randn(p + 1), np.zeros(p + 1), np.zeros(p + 1)  #
    for epoch in range(n_epoches):
        for _ in range(n // batch_size):
            indexes = np.random.permutation(n)[:batch_size]
            batch_X, batch_y = X[indexes], y[indexes]
            loss = ((batch_X @ omega - batch_y) ** 2).mean() + mu * (omega[1:] ** 2).sum()
            # 惩罚项关于截距的梯度为0,因此第二项首元素为0
            grad = 2 / batch_size * batch_X.T @ (batch_X @ omega - batch_y) + 2 * mu * np.hstack(([0], omega[1:]))
            S = rho1 * S + (1 - rho1) * grad  # Momentum
            M = rho2 * M + (1 - rho2) * grad ** 2  # RMSProp
            rho1k, rho2k = rho1k * rho1, rho2k * rho2
            S_hat, M_hat = S / (1 - rho1k), M / (1 - rho2k)  # 修正一阶矩，二阶矩
            omega -= lr / (np.sqrt(M_hat) + eps) * S_hat  # eps 防止下溢
        print(f'epoch {epoch}, loss {loss}')
    return omega


if __name__ == '__main__':
    from sklearn.linear_model import Lasso

    np.random.seed(12)
    # y = 10 + 5 x1 + 3 x2
    n = 10000
    x_1 = np.linspace(-10, 10, n)
    x_2 = np.linspace(-10, 10, n)
    y = 10 + 5 * x_1 + 3 * x_2
    X = np.hstack((x_1[:, None] + np.random.normal(0, 2, (n, 1)), x_2[:, None] + np.random.normal(0, 1, (n, 1))))

    model = Lasso(alpha=25)  # sklearn中 均方误差除以了2,对应惩罚项也要除以2
    model.fit(X.copy(), y.copy())
    print(model.intercept_, model.coef_)

    omega = LASSO(X, y, 50, 50)
    print(omega)

    from sklearn.linear_model import Ridge

    np.random.seed(12)
    # y = 10 + 5 x1 + 3 x2
    n = 10000
    x_1 = np.linspace(-10, 10, n)
    x_2 = np.linspace(-10, 10, n)
    y = 10 + 5 * x_1 + 3 * x_2
    X = np.hstack((x_1[:, None] + np.random.normal(0, 2, (n, 1)), x_2[:, None] + np.random.normal(0, 1, (n, 1))))

    model = Ridge(alpha=5000, fit_intercept=True)
    model.fit(X, y)
    print(model.intercept_, model.coef_)

    omega = ridge_regression(X, y, 5000)
    print(omega)
    omega = ridge_regression_Adam(X, y, 5000 / n, 1e-2, 0.9, 0.999, 100, 64)  # 均方误差而不是残差平方和,因此对应惩罚项也要除以n
    print(omega)
