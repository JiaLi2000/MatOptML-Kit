import numpy as np


def simplex(A, b, c):  # 标准型LP: min c'x s.t. Ax = b,x>=0
    m, n = A.shape  # 这里的n是标准型的n，尚未添加人工变量
    A, c = np.hstack((A, np.eye(m))), np.hstack((c, np.full(m, 1e8)))  # 1.人工变量法之大M法构造单位初始基可行解
    B, x = np.arange(n, n + m), np.zeros(n + m)  # 通过基索引B定位当前单位基
    x[B], sigma = b, c - c[B] @ A  # 计算初始单位基可行解，计算所有变量的检验数
    while True:
        if np.all(sigma >= 0):  # 2.最优性检验:检验数均非负，则目标值无法下降,已达最小
            if np.any(x[n:] != 0):  # 若最优解存在人工变量非0，则原问题无可行解
                print('原问题无可行解')
                break
            else:
                print(f'最优值{c @ x}, 最优解{x[:n]}')  # 若最优解的人工变量均非0，则解的非人工变量部分即原问题最优解
                break
        else:  # 一入一出进行基变换
            k = np.argmin(sigma)  # 最小检验数原则确定入基变量
            if np.all(A[:, k] <= 0):
                print('原问题无界解')
                break
            else:
                posi = np.where(A[:, k] > 0)[0]  # 只比较A[:, k] > 0 的元素对应的比值
                l = posi[np.argmin(b[posi] / A[posi, k])]  # 最小比值法确定出基变量，外侧加posi是为了取行号
            b[l] /= A[l, k]  # 将含入基变量的逆单位阵进行单位化形成新的单位可行基，注意运算先后顺序
            A[l, :] /= A[l, k]
            for i in range(m):
                if i == l:
                    continue
                else:
                    b[i] -= A[i, k] * b[l]
                    A[i, :] -= A[i, k] * A[l, :]
            B[l], x[:] = k, 0  # 更新基索引
            x[B], sigma = b, c - c[B] @ A  # 可行基为单位阵时，基可行解的基变量取值即为右端项
            # print(f'当前解{c @ x}')


if __name__ == '__main__':
    # 原问题
    # max -3x1 + x3
    # s.t. x1 + x2 + x3 <= 4
    #     -2x1+x2-x3 >=1
    # 3x2 + x3 = 9
    # x1,x2,x3 >= 0

    # 化为标准型
    A = np.array([[1, 1, 1, 1, 0], [-2, 1, -1, 0, -1], [0, 3, 1, 0, 0]])
    c = np.array([3, 0, -1, 0, 0])
    b = np.array([4, 1, 9.0])
    simplex(A, b.copy(), c)

    import scipy.optimize as so
    print(so.linprog(c,A_eq=A, b_eq=b, method='highs'))
