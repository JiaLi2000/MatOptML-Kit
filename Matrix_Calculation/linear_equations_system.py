import numpy as np


def forward_sub(L, b):  # 前代法：L为n阶可逆下三角阵，O(n^2)
    n = b.shape[0]
    b[0] = b[0] / L[0, 0]  # 首行方程的解，同时用b存储x
    for i in range(1, n):
        b[i] = (b[i] - L[i, :i] @ b[:i]) / L[i, i]  # 常数项减去左端除以系数
    return b


def backward_sub(U, b):  # 回代法：U为n阶可逆上三角阵，O(n^2)
    n = b.shape[0]
    b[n - 1] = b[n - 1] / U[n - 1, n - 1]  # 末行方程的解，同时用b存储x
    for i in range(n - 2, -1, -1):
        b[i] = (b[i] - U[i, i + 1:] @ b[i + 1:]) / U[i, i]  # 常数项减去右端除以系数
    return b


def LU(A):  # LU分解：A为n阶方阵且顺序主子式均非0, O(n^3)
    n = A.shape[0]
    for i in range(n - 1):
        A[i + 1:, i] = A[i + 1:, i] / A[i, i]  # 计算针对各列的高斯变换的高斯向量，并就地存储至A(该列消元后,主元下方均为0)
        A[i + 1:, i + 1:] -= A[i + 1:, i][:, None] @ A[i, i + 1:][None, :]  # 对剩余的子式同步执行相应的高斯变换
    L = np.tril(A)  # 由高斯变换的性质3,L由各次消元的高斯向量构成
    np.fill_diagonal(L, 1)
    return L, np.triu(A)  # n-1次高斯变换后，A变为上三角U


def PLU(A):  # PA=LU分解：A为n阶可逆方阵, O(n^3)
    n = A.shape[0]
    P = np.eye(n)
    for i in range(n - 1):
        maxp = np.argmax(np.abs(A[i:, i])) + i  # 选择当前子式(防止已消去元素变回非0)首列中最大元素作为列主元
        P[[i, maxp], :] = P[[maxp, i], :]  # 记录每次置换的置换矩阵P_i,最终得到P = P_{n-1}...P_1
        A[[i, maxp], :] = A[[maxp, i], :]  # 执行主元置换，注意这同时对高斯向量l_k执行了后续的置换
        A[i + 1:, i] = A[i + 1:, i] / A[i, i]  # 计算针对各列的高斯变换的高斯向量，并就地存储至A(该列消元后,主元下方均为0)
        A[i + 1:, i + 1:] -= A[i + 1:, i][:, None] @ A[i, i + 1:][None, :]  # 对剩余的子式同步执行相应的高斯变换
    L = np.tril(A)  # 由高斯变换的性质4,L由各次消元的高斯向量做同等后续置换得到
    np.fill_diagonal(L, 1)
    return P, L, np.triu(A)  # n-1次置换+高斯变换后，A变为上三角U


def solve(A, b):
    P, L, U = PLU(A)
    y = forward_sub(L, P @ b)
    x = backward_sub(U, y)
    return x


if __name__ == '__main__':
    np.random.seed(10)
    A = np.random.random((5, 5)) * 10
    # 前代
    L = np.tril(A)
    np.fill_diagonal(L, 2)
    print(L)
    b = np.array([5.0, 3, 7, 6, 2])
    print(forward_sub(L, b.copy()))
    print(np.linalg.solve(L, b.copy()))
    # 回代
    U = np.triu(A)
    np.fill_diagonal(U, 2)
    print(U)
    b = np.array([5.0, 3, 7, 6, 5])
    print(backward_sub(U, b.copy()))
    print(np.linalg.solve(U, b.copy()))
    # LU
    A = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 10]])
    print(LU(A.copy()))
    from scipy.sparse.linalg import splu

    # slu = splu(A, diag_pivot_thresh=0)  # 等效于不执行置换的LU分解
    # print(slu.L.toarray(), slu.U.toarray())
    # PLU
    print('PLU')
    import scipy.linalg as sl

    A = np.array([[1, 2.0, 0], [1, 2, 1], [0, 2, 0]])
    P, L, U = PLU(A.copy())
    print(P.T, '\n', L, '\n\n', U)
    print('-----')
    p, L, U = sl.lu(A.copy())
    print(p, L, U)

    # linear system
    A = np.random.random((5, 5)) * 10
    b = np.array([5.0, 3, 7, 6, 2])
    print(solve(A.copy(), b))
    print(np.linalg.solve(A, b))
