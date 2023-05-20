import numpy as np


def forward_sub(L, b):
    n = b.shape[0]
    b[0] = b[0] / L[0, 0]
    for i in range(1, n):
        b[i] = (b[i] - L[i, :i] @ b[:i]) / L[i, i]
    return b


def backward_sub(U, b):
    n = b.shape[0]
    b[n - 1] = b[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        b[i] = (b[i] - U[i, i + 1:] @ b[i + 1:]) / U[i, i]
    return b


def LU(A):
    n = A.shape[0]
    for i in range(n - 1):
        A[i + 1:, i] = A[i + 1:, i] / A[i, i]  # 画家算法,计算L的第i列, 即高斯变换L_i非01元素取反
        A[i + 1:, i + 1:] -= A[i + 1:, i][:, None] @ A[i, i + 1:][None, :]  # 对剩余的子式执行相应的初等行变换
    L = np.tril(A)
    np.fill_diagonal(L, 1)
    return L, np.triu(A)

def PLU(A):
    n = A.shape[0]
    p = np.zeros(n-1)
    for i in range(n - 1):
        maxp = np.argmax(np.abs(A[i:, i])) + i
        p[i] = maxp
        temp = A[maxp,:].copy()
        A[maxp, :] = A[i,:]
        A[i, :] = temp
        A[i + 1:, i] = A[i + 1:, i] / A[i, i]  # 画家算法,计算L的第i列, 即高斯变换L_i非01元素取反
        A[i + 1:, i + 1:] -= A[i + 1:, i][:, None] @ A[i, i + 1:][None, :]  # 对剩余的子式执行相应的初等行变换
    L = np.tril(A)
    np.fill_diagonal(L, 1)

    return p, L, np.triu(A)


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
        slu = splu(A, diag_pivot_thresh=0)  # 等效于不执行置换的LU分解
        print(slu.L.toarray(), slu.U.toarray())
        # PLU
        print('PLU')
        import scipy.linalg as sl
        A = np.array([[2.0, 5, 2, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
        p,L,U = PLU(A.copy())
        print(p,'\n',L,U)
        print('-----')
        p,L,U = sl.lu(A.copy())
        print(p,L,U)
