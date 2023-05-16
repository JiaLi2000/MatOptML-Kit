import numpy as np
import scipy.linalg as sl
import time


def power_method(A: np.array, mu: float = 0, max_iters: int = 100, eps: float = 1e-3) -> (np.array, np.array):
    '''
    幂法计算A的主特征值及主特征向量
    :param A: A可对角化,且主特征值的模严格大
    :param mu: 位移加速
    :param max_iters: 最大迭代次数
    :param eps: 特征值绝对误差
    :return:
        主特征值,主特征向量
    '''
    n = A.shape[0]
    x = np.random.random(n)
    pre_lambda, cur_lambda = 0, 0
    I = np.eye(n)
    iter = 0
    while True:
        y = (A - mu * I)  @ x
        x = y / ((y ** 2).sum() ** 0.5)
        pre_lambda = cur_lambda
        cur_lambda = x[None, :] @ A @ x
        iter += 1
        if abs(cur_lambda - pre_lambda) < eps or iter > max_iters:
            break
    return cur_lambda, x


def inverse_power_method(A: np.array, mu: float = 0, max_iters: int = 100, eps: float = 1e-3) -> (np.array, np.array):
    '''
    反幂法计算A的接近mu的特征值及主特征向量
    :param A: nxn实对称矩阵
    :param mu: 待估计特征值
    :param max_iters: 最大迭代次数
    :param eps: 特征值绝对误差
    :return:
        接近mu的特征值及对应的特征向量
    '''
    n = A.shape[0]
    x = np.random.random(n)
    pre_lambda, cur_lambda = 0, 0
    I = np.eye(n)
    iter = 0
    while True:
        y = sl.solve(A - mu * I, x)
        x = y / ((y ** 2).sum() ** 0.5)
        pre_lambda = cur_lambda
        cur_lambda = x[None, :] @ A @ x
        iter += 1
        if abs(cur_lambda - pre_lambda) < eps or iter > max_iters:
            break
    return cur_lambda, x


def gradient_descent(fun, theta: np.array, lr: float = 1e-3, max_iters: int = 500, eps: float = 1e-3,  **paras) -> (np.array, np.array):
    '''

    :param fun: 被优化的目标函数
    :param theta: 参数
    :param lr: 学习率
    :param max_iters: 最大迭代次数
    :param eps: 目标函数相对误差阈值
    :param paras: 目标函数的其他自变量
    :return:
        优化停止时的目标函数值,参数值
    '''
    pre_fval = 0
    iter = 0
    while True:
        cur_fval, grad = fun(theta=theta, **paras)
        theta -= lr * grad
        print(f'iter {iter}, loss {cur_fval}, grad{grad}')
        if iter > max_iters or abs(cur_fval - pre_fval) < eps:
            break
        pre_fval = cur_fval
        iter += 1
    return cur_fval, theta



if __name__ == '__main__':
    np.random.seed(3)
    A = np.random.random((5,5)) * 10
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
    eigvalue, eigvector = inverse_power_method(A,0)
    t2 = time.perf_counter()
    print(eigvalue, eigvector)
    print(t2 - t1)

    print(sl.eig(A))