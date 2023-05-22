import numpy as np


def power_method(A, mu=0, max_iters=100, eps=1e-3):
    n, iter = A.shape[0], 0
    x, I = np.random.random(n), np.eye(n)
    pre_lambda, cur_lambda = 0, 1
    while abs(cur_lambda - pre_lambda) >= eps and iter <= max_iters:
        pre_lambda, iter = cur_lambda, iter + 1
        y = (A - mu * I) @ x  # 幂迭代, mu用于移位加速
        x = y / ((y ** 2).sum() ** 0.5)  # 标准化防止上下溢出
        cur_lambda = x[None, :] @ A @ x  # 由单位特征向量计算对应特征值
    return cur_lambda, x


def inverse_power_method(A, mu=0, max_iters=100, eps=1e-3):
    n, iter = A.shape[0], 0
    x, I = np.random.random(n), np.eye(n)
    pre_lambda, cur_lambda = 0, 1
    while abs(cur_lambda - pre_lambda) >= eps and iter <= max_iters:
        pre_lambda, iter = cur_lambda, iter + 1
        y = sl.solve(A - mu * I, x)  # 用解方程组代替求逆, mu是待估计特征值
        x = y / ((y ** 2).sum() ** 0.5)  # 标准化防止上下溢出
        cur_lambda = x[None, :] @ A @ x  # 由单位特征向量计算对应特征值
    return cur_lambda, x


if __name__ == '__main__':
    import scipy.linalg as sl
    import time

    np.random.seed(4)
    A = np.random.random((5, 5)) * 10
    A = A.T @ A

    t1 = time.perf_counter()
    eigvalue, eigvector = power_method(A)
    t2 = time.perf_counter()
    print(eigvalue, eigvector)
    print(t2 - t1)

    t1 = time.perf_counter()
    eigvalue, eigvector = power_method(A, 100)
    t2 = time.perf_counter()
    print(eigvalue, eigvector)
    print(t2 - t1)

    print('----------')
    t1 = time.perf_counter()
    eigvalue, eigvector = inverse_power_method(A, 0)
    t2 = time.perf_counter()
    print(eigvalue, eigvector)
    print(t2 - t1)

    print(sl.eigh(A))
