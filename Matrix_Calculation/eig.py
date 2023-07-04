import numpy as np


def power_method(A, mu=0, T=100, eps=1e-3):  # 位移加速的幂法： A为n阶可对角化方阵且谱半径严格大。O(Tn^2)
    n, t = A.shape[0], 0
    xi, I = np.random.random(n), np.eye(n)
    pre, cur = 0, 1
    while abs(cur - pre) >= eps and t <= T:
        pre, t = cur, t + 1
        xi = (A - mu * I) @ xi  # mu为位移加速,实际上是计算矩阵B:= A - mu*I的主特征值, B的收敛速度更快
        xi = xi / ((xi ** 2).sum() ** 0.5)  # 标准化特征向量中间结果，防止其元素上下溢出
        cur = xi[None, :] @ A @ xi  # 由单位特征向量计算对应特征值
    return cur, xi  # 返回A的主特征值cur及主特征向量xi


def deflated_power_method(A, k, mu=0, T=200, eps=1e-5):  # 降阶的幂法: A为n阶实对称矩阵且top-k特征值的模互不相同。O(Tkn^2)
    n = A.shape[0]
    eigvalues, eigvectors = np.zeros(k), np.zeros((n, k))
    for i in range(k):
        eigvalues[i], eigvectors[:, i] = power_method(A, mu, T, eps)  # 对将阶后的矩阵执行幂法,其主特征对对应原矩阵的特征对
        A -= eigvalues[i] * eigvectors[:, i][:, None] @ eigvectors[:, i][None, :]  # 通过减去正交谱分解的一项完成降阶
    return eigvalues, eigvectors


def inversed_power_method(A, mu=0.0, T=100, eps=1e-3):  # 反幂法：A-mu*I可逆,位移mu与待估计特征值接近. O(Tn^3)
    n, t = A.shape[0], 0
    xi, I = np.random.random(n), np.eye(n)
    pre, cur = 0, 1
    while abs(cur - pre) >= eps and t <= T:
        pre, t = cur, t + 1
        xi = np.linalg.solve(A - mu * I, xi)  # 用解方程组代替求逆(系数矩阵固定，因此可用LU分解加速)
        xi = xi / ((xi ** 2).sum() ** 0.5)  # 标准化特征向量中间结果，防止其元素上下溢出
        cur = xi[None, :] @ A @ xi  # 由单位特征向量计算对应特征值
    return cur, xi


def QR_householder(A):  # QR分解：A \in R^{m,n}，m >= n, O(mn^2)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - 1):  # 对A的子矩阵依次Householder消元,每次将子式首列化为alpha*e_1
        alpha = (A[i:, i] ** 2).sum() ** 0.5  # Householder变换后e1的系数, 符号决定R矩阵对角线的符号
        omega = A[i:, i] - alpha * np.eye(1, m - i)[0]  # A[i:, i]表示子矩阵首列
        # 计算对当前子矩阵首列Householder变换的Householder向量,当子矩阵首列天然为alpha*e_1时,分母为0
        omega = omega / (omega ** 2).sum() ** 0.5 if (omega ** 2).sum() ** 0.5 != 0 else omega
        sub_H = np.eye(m - i) - 2 * omega[:, None] @ omega[None, :]  # 由Householder向量得到对子矩阵的Householder变换
        H = np.block([[np.eye(i), np.zeros((i, m - i))],
                      [np.zeros((m - i, i)), sub_H]]) if i > 0 else sub_H  # 单位扩充的Householder变换
        A, Q = H @ A, Q @ H  # 将扩充后的Householder变换作用在A上(对子矩阵行变换而不是仅其首列),累乘Householder变换得到Q矩阵
    return Q, np.triu(A)


def QR_schmidt(A):  # QR分解：A \in R^{m,n}，m >= n, rank(A) = n, O(mn^2)
    m, n = A.shape
    Q, R = np.eye(m), np.zeros((n, n))  # Q初始化为单位阵,直接对m > n的列添加了标准正交向量增广
    Q[:, :n] = A  # schmidt公式左端为原向量
    for j in range(n):  # 依次计算标准正交向量q_j，并在计算q_{j+1}时使用
        for i in range(j):  # 正交向量q_j被表示的系数为R的列
            R[i, j] = A[:, j] @ Q[:, i]  # R_{ij} = q_i^\top a_j / q_i^\top q_i,边正交化边单位化，因此系数的分母为1
            Q[:, j] -= R[i, j] * Q[:, i]  # schmidt公式右端连减
        R[j, j] = (Q[:, j] @ Q[:, j]) ** 0.5  # 单位正交向量的系数才是R中的元素
        Q[:, j] /= R[j, j]  # 对正交向量单位化
    return Q, R


def QR_iteration(A, T=100, eps=1e-1):  # QR迭代: A为n阶方阵， O(Tn^3)
    n, t = A.shape[0], 0
    pre, cur = np.zeros(n), np.full(n, 100)
    while ((cur - pre) ** 2).sum() ** 0.5 >= eps and t <= T:  # 一定条件下对角线收敛至特征值,以下元素为0,这里仅考虑对角线
        pre, t = cur, t + 1
        Q, R = QR_householder(A)
        A = R @ Q  # A_{k+1}收敛至实Schur补，且正交相似于A
        cur = np.diagonal(A)
    return np.diagonal(A).copy()  # 实Schur补的主对角线为A的特征值


def eigs(A, T=10, eps=1e-1):  # 对特征值绝对值互异的实对称矩阵的特征分解, O(Tn^3)
    n = A.shape[0]
    eigvalues = QR_iteration(A.copy(), T, eps)
    eigvectors = np.zeros((n, n))
    for i in range(n):
        eigvalues[i], eigvectors[:, i] = inversed_power_method(A, eigvalues[i] + eps, 100 * T,
                                                               1e-2 * eps)  # 为真实特征值加扰动,否则反幂法对应矩阵不可逆
    eigvectors, _ = QR_schmidt(eigvectors)  # schmidt 正交化, 当特征值非互异时,反幂法失效，只能用其他方法求特征向量
    desc_indexes = np.argsort(eigvalues)[::-1]  # 按特征值的值降序排列
    return eigvalues[desc_indexes], eigvectors[:, desc_indexes]


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

    t1 = time.perf_counter()
    eigvalue, eigvector = deflated_power_method(A.copy(), 3)
    t2 = time.perf_counter()
    print(eigvalue, '\n', eigvector)
    print(t2 - t1)

    print('----------')
    t1 = time.perf_counter()
    eigvalue, eigvector = inversed_power_method(A, 0)
    t2 = time.perf_counter()
    print(eigvalue, eigvector)
    print(t2 - t1)

    print(sl.eigh(A))
    A = np.array([[0, 3, 1, 0], [0, 4, -2, 0], [2, 1, 1, 0], [0, 0, 0, 0]])
    Q, R = QR_householder(A)
    print(np.round(Q, 4), '\n', np.round(R, 4))
    import scipy.linalg as sl

    np.random.seed(100)
    A = np.random.random((5, 5)) * 10
    A = A.T @ A
    print(sl.qr(A))
    print(np.round(QR_iteration(A, T=500), 1))
    print(sl.eigh(A))

    A = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0]])
    Q, R = QR_schmidt(A)
    print(Q)
    print('------', '\n', R)

    A = np.array([[1, 2, 3, 4, 5],
                  [2, 3, 4, 5, 6],
                  [3, 4, 5, 6, 7],
                  [4, 5, 6, 7, 8],
                  [5, 6, 7, 8, 9.0]])
    A = np.triu(A)
    mu = inversed_power_method(A, mu=5.01, T=100, eps=1e-3)
    print(mu)
    print(np.linalg.eig(A))

    print('-------')
    np.random.seed(8)
    A = np.random.random((5, 5)) * 10
    A = A.T @ A
    print(sl.eigh(A))
    QR_iteration(A, T=100, eps=1e-10)
    print(eigs(A, 100, 1e-1))
