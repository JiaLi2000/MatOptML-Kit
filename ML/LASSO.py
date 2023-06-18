import numpy as np


def LASSO_BCD(X, y, T, mu):  # min 1/n ||Xw + b -y||_2^2 + \mu ||w||_1 坐标下降法求LASSO问题
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

    omega = LASSO_BCD(X, y, 50, 50)
    print(omega)
