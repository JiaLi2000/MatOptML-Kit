import numpy as np
import eig


def OLS(A, b, method):  # A为mxn列满秩矩阵, 求最小二乘问题 min ||Ax-b||^2
    n, p = A.shape  # 这里的OLS与线性回归本质一样,区别在于这里用方程组的角度描述，只用精确算法求解
    if method == 'inv':
        return np.linalg.inv(A.T @ A) @ A.T @ b  # 正规方程组直接求逆,也可以用LU分解
    elif method == 'qr':
        Q, R = np.linalg.qr(A, mode='complete')  # QR分解,这里R是nxp的
        return np.linalg.solve(R[:p, :], (Q.T @ b)[:p])
    elif method == 'svd':
        U, S, VT = np.linalg.svd(A)  # Sigma是一维向量
        return np.linalg.solve(np.diag(S) @ VT, U[:, :p].T @ b)


def null(A):  # 计算mxn矩阵A为的零空间 Nul(A), O(n^3)。注意：这里使用的QR分解要求m >= n
    m, n = A.shape
    Q, R = eig.QR_householder(A)  # R为上三角,(R 0)^\top 为行阶梯,与A有相同的行最简
    R = np.round(R, 4)  # 规约非常接近0的浮点数
    free_indexes = np.argwhere(R.diagonal() == 0).flatten()  # 取对角线为0元素对应的列为自由列
    if len(free_indexes) == 0:  # 列满秩
        return None
    for j in range(n):  # 将行阶梯向上消元得到行最简
        if j not in free_indexes:
            R[j, :] /= R[j, j]  # 选择R[j, j]作为主元,并将其行变换为1
            R[:j, :] -= R[:j, j][:, None] @ R[j, :][None, :]  # 通过行变换将主元R[j, j]上方元素置为0
    R *= -1
    np.fill_diagonal(R, 1)  # 行最简后,将自由向量的对角位置置1,其他位置取反，得到自由基
    bases = R[:, free_indexes]
    normalized_bases, _ = eig.QR_householder(bases)  # 对自由基正交化
    return normalized_bases[:, :len(bases) - 1]


def svd(A):  # 计算mxn矩阵的SVD分解, 这里假设n > m, O(n^3)
    m, n = A.shape
    eigvalues, V = eig.eigs(A.T @ A)  # A.T @ A的n个正交特向构成V,
    sigma = np.sqrt(eigvalues)[:m]  # A.T @ A的特征值开方构成A的奇异值
    Sigma = np.block([np.diag(sigma), np.zeros((m, n - m))])  # n > m的矩阵,从奇异值补全为mxn的对角矩阵
    positive_index = np.argwhere(sigma > 0).flatten()  # 正奇异值对应的下标，用于构建U1
    U1 = A @ V[:, positive_index] / sigma[positive_index]  # U1为Col(A)的标准正交基
    U2 = null(A.T)  # U2为Nul(A^\top)的标准正交基
    U = np.hstack([U1, U2]) if U2 is not None else U1
    return U, Sigma, V.T


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression

    np.random.seed(12)
    # y = 10 + 5 x1 + 3 x2
    n = 10000
    x_1 = np.linspace(-10, 10, n)
    x_2 = np.linspace(-10, 10, n)
    y = 10 + 5 * x_1 + 3 * x_2
    X = np.hstack((x_1[:, None] + np.random.normal(0, 2, (n, 1)), x_2[:, None] + np.random.normal(0, 1, (n, 1))))

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    print(model.coef_)

    omega = OLS(X, y, 'inv')
    print(omega)
    omega = OLS(X, y, 'qr')
    print(omega)
    omega = OLS(X, y, 'svd')
    print(omega)

    import scipy.linalg as sl

    np.random.seed(12)
    w = np.random.rand(5)
    A = np.outer(w, w)
    print(sl.null_space(A))
    ns = null(A)
    print(null(A))

    print("--------")
    A = np.random.random((4, 8))
    U, S, VT = np.linalg.svd(A)
    print(U, S, VT)
    U, S, VT = svd(A)
    print(U, S, VT)
