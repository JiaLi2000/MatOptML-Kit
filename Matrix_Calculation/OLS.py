import numpy as np


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

# 这里可以补充一下SVD算法和求逆算法



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
