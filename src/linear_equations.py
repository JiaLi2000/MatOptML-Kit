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
