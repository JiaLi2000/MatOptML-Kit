import numpy as np


def power_method(A, mu=0, T=100, eps=1e-3):  # 幂法： A为n阶可对角化方阵且主特征值的模严格大。O(Tn^2)
    n, t = A.shape[0], 0
    xi, I = np.random.random(n), np.eye(n)
    pre, cur = 0, 1
    while abs(cur - pre) >= eps and t <= T:
        pre, t = cur, t + 1
        xi = (A - mu * I) @ xi  # mu为位移加速,实际上是计算矩阵B:= A - mu*I的主特征值, B的收敛速度更快
        xi = xi / ((xi ** 2).sum() ** 0.5)  # 标准化特征向量中间结果，防止其元素上下溢出
        cur = xi[None, :] @ A @ xi  # 由单位特征向量计算对应特征值
    return cur, xi  # 返回A的主特征值cur及主特征向量xi


def deflated_power_method(A, k, mu=0, T=200, eps=1e-5):  # 幂法+收缩/降阶: A为n阶实对称矩阵且topk特征值互不相同。O(Tkn^2)
    n = A.shape[0]
    eigvalues, eigvectors = np.zeros(k), np.zeros((n, k))
    for i in range(k):
        eigvalues[i], eigvectors[:, i] = power_method(A, mu, T, eps)  # 对将阶后的矩阵执行幂法,其主特征对对应原矩阵的特征对
        A -= eigvalues[i] * eigvectors[:, i][:, None] @ eigvectors[:, i][None, :]  # 通过减去正交谱分解的一项完成降阶
    return eigvalues, eigvectors


def inversed_power_method(A, mu=0, T=100, eps=1e-3):  # 反幂法：A-mu*I可逆,位移mu与待估计特征值接近. O(Tn^3)
    n, t = A.shape[0], 0
    xi, I = np.random.random(n), np.eye(n)
    pre, cur = 0, 1
    while abs(cur - pre) >= eps and t <= T:
        pre, t = cur, t + 1
        xi = np.linalg.solve(A - mu * I, xi)  # 用解方程组代替求逆(系数矩阵固定，因此可用LU分解加速)
        xi = xi / ((xi ** 2).sum() ** 0.5)  # 标准化特征向量中间结果，防止其元素上下溢出
        cur = xi[None, :] @ A @ xi  # 由单位特征向量计算对应特征值
    return cur, xi


def QR(A):  # QR分解：A \in R^{m,n}，m >= n, O(m^2n)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - 1):
        alpha = (A[i:, i] ** 2).sum() ** 0.5  # Householder变换后e1的系数, 符号决定R矩阵对角线的符号
        omega = A[i:, i] - alpha * np.eye(1, m - i)[0]
        omega = omega / (omega ** 2).sum() ** 0.5  # 计算对当前子矩阵首列Householder变换的Householder向量
        sub_H = np.eye(m - i) - 2 * omega[:, None] @ omega[None, :]  # 由Householder向量得到Householder变换
        H = np.block([[np.eye(i), np.zeros((i, m - i))],
                      [np.zeros((m - i, i)), sub_H]]) if i > 0 else sub_H  # 单位扩充的Householder变换
        A, Q = H @ A, Q @ H  # 将扩充后的Householder变换作用在A上,累乘Q矩阵
    return Q, np.triu(A)


def QR_iteration(A, T=20):
    for t in range(T):
        Q, R = QR(A)
        A = R @ Q
    return A


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
    Q, R = QR(A)
    print(np.round(Q, 4), '\n', np.round(R, 4))
    import scipy.linalg as sl

    print(sl.qr(A))
    print(np.round(QR_iteration(A,11),4))
    print(sl.eigh(A))