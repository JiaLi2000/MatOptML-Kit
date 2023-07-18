import numpy as np


def lr_SGD(X, y, lr, gamma, n_epoches, batch_size):  # 带动量的小批量梯度下降，gamma=0时退化为小批量梯度下降
    n, p = X.shape
    np.random.seed(43)
    X = np.hstack((np.ones((n, 1)), X))
    omega, delta = np.random.randn(p + 1), 0  # delta为历史变化量的指数加权平均
    for epoch in range(n_epoches):
        for _ in range(n // batch_size):
            indexes = np.random.permutation(n)[:batch_size]
            batch_X, batch_y = X[indexes], y[indexes]
            loss = ((batch_X @ omega - batch_y) ** 2).mean()
            grad = 2 / batch_size * batch_X.T @ (batch_X @ omega - batch_y)
            delta = gamma * delta - lr * grad  # gamma 为动量系数，利用当前点的梯度对上次迭代的变化量进行纠正
            omega += delta
        print(f'epoch {epoch}, loss {loss}')  # 简单起见,输出epoch最后一个批次的损失
    return omega


def lr_NAG(X, y, lr, gamma, n_epoches, batch_size):  # Nesterov’s Accelerated Gradient
    n, p = X.shape
    np.random.seed(43)
    X = np.hstack((np.ones((n, 1)), X))
    omega, delta = np.random.randn(p + 1), 0
    for epoch in range(n_epoches):
        for _ in range(n // batch_size):
            indexes = np.random.permutation(n)[:batch_size]
            batch_X, batch_y = X[indexes], y[indexes]
            loss = ((batch_X @ omega - batch_y) ** 2).mean()
            omega_ahead = omega + gamma * delta  # 前瞻点，用作下个点的估计
            grad = 2 / batch_size * batch_X.T @ (batch_X @ omega_ahead - batch_y)
            delta = gamma * delta - lr * grad  # 利用前瞻点的梯度对上次迭代的变化量进行纠正
            omega += delta  # 注意在原点上更新，而不是前瞻点
        print(f'epoch {epoch}, loss {loss}')
    return omega


def lr_AdaGrad(X, y, lr, n_epoches, batch_size):  # AdaGrad
    n, p = X.shape
    np.random.seed(43)
    X, eps = np.hstack((np.ones((n, 1)), X)), 1e-8
    omega, G = np.random.randn(p + 1), np.zeros(p + 1)  # G为历史梯度的累积平方和
    for epoch in range(n_epoches):
        for _ in range(n // batch_size):
            indexes = np.random.permutation(n)[:batch_size]
            batch_X, batch_y = X[indexes], y[indexes]
            loss = ((batch_X @ omega - batch_y) ** 2).mean()
            grad = 2 / batch_size * batch_X.T @ (batch_X @ omega - batch_y)
            G += grad ** 2  # 历史梯度的累积平方和
            omega -= lr * grad / (np.sqrt(G) + eps)  # eps 防止下溢
        print(f'epoch {epoch}, loss {loss}')
    return omega


def lr_RMSProp(X, y, lr, rho, n_epoches, batch_size):  # RMSProp
    n, p = X.shape
    np.random.seed(43)
    X, eps = np.hstack((np.ones((n, 1)), X)), 1e-8
    omega, G = np.random.randn(p + 1), np.zeros(p + 1)  # G为历史梯度的累积平方和的指数加权平均
    for epoch in range(n_epoches):
        for _ in range(n // batch_size):
            indexes = np.random.permutation(n)[:batch_size]
            batch_X, batch_y = X[indexes], y[indexes]
            loss = ((batch_X @ omega - batch_y) ** 2).mean()
            grad = 2 / batch_size * batch_X.T @ (batch_X @ omega - batch_y)
            G = rho * G + (1 - rho) * grad ** 2  # 对历史梯度的累积平方和进行指数加权平均
            omega -= lr * grad / (np.sqrt(G) + eps)  # eps 防止下溢
        print(f'epoch {epoch}, loss {loss}')
    return omega


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

    omega = lr_AdaGrad(X, y, 5e-1, 12, 64)  # AdaGrad
    print(omega)

    omega = lr_RMSProp(X, y, 5e-1, 0.9, 12, 64)  # RMSProp
    print(omega)

    omega = lr_Adam(X, y, 5e-1, 0.9, 0.999, 12, 64)  # Adam
    print(omega)
