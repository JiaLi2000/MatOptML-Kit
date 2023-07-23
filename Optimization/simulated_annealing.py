import numpy as np


def sa(x0, T, alpha, L, v, w, b):  # 模拟退火算法求解01背包问题
    v = -v  # 背包问题为最大化问题,转为最小化
    x_best, x = x0, x0
    while T > 0.1:
        for _ in range(L):  # 每个温度下状态转移一定次数到达平衡态
            idx = np.random.randint(len(x))
            x_prime = x.copy()
            x_prime[idx] = 1 - x_prime[idx]  # 随机翻转一位来生成新解
            if w @ x_prime > b:  # 非可行则重新生成新解
                continue
            if v @ x_prime < v @ x:  # 新状态能量更低则转移
                x = x_prime
                if v @ x < v @ x_best:  # 新状态比当前最优能量还低则更新
                    x_best = x
                continue
            if np.random.random(1) < np.exp(- (v @ x_prime - v @ x) / T):  # Metropolis准则
                x = x_prime
        T *= alpha  # 冷却温度
    return x_best, -v @ x_best  # 注意负号取回来


if __name__ == '__main__':
    v = np.array([8, 11, 6, 4])
    w = np.array([5, 7, 4, 3])
    b = 14
    x0 = np.array([0, 1, 0, 0])
    x_best, f_best = sa(x0, 10, 0.99, 20, v, w, b)
    print(x_best, f_best)
