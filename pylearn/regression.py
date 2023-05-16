import numpy as np
import optim
import scipy.linalg as sl
import time
from sklearn.linear_model import LinearRegression


def linear_regression(X: np.array, y: np.array, solution: str = 'exact', **paras):
    n, p = X.shape
    aug_X = np.hstack((np.ones((n, 1)), X))
    if solution == 'exact':
        return sl.inv(aug_X.T @ aug_X) @ aug_X.T @ y
    else:
        np.random.seed(43)
        theta = np.random.random(p + 1)

        def loss(X, theta, y):
            temp = X @ theta - y
            return (temp ** 2).mean(), 2 * (X.T @ temp).mean()

        fval, theta = optim.gradient_descent(loss, theta, paras['lr'], paras['max_iters'], paras['eps'], X=aug_X, y=y)
        return theta


if __name__ == '__main__':
    np.random.seed(10)
    # y = 10 + 5 x1 + 3 x2
    n = 1000
    x_1 = np.linspace(-10, 10, n)
    x_2 = np.linspace(-10, 10, n)
    y = 10 + 5 * x_1 + 3 * x_2
    X = np.hstack((x_1[:, None] + np.random.normal(0, 2, (n, 1)), x_2[:, None] + np.random.normal(0, 1, (n, 1))))



    n, p = X.shape
    aug_X = np.hstack((np.ones((n, 1)), X))

    beta = linear_regression(X, y, solution='GD', lr=8e-6, max_iters=1000, eps=1e-20)
    print(beta)
    print(((aug_X @ beta - y) ** 2).mean())



    beta = linear_regression(X, y, solution='exact')
    print(beta)
    print(((aug_X @ beta - y) ** 2).mean())
    # model = LinearRegression()
    # model.fit(X,y)
    # print(model.coef_, model.intercept_)