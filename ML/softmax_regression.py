import numpy as np


def softmax(x):
    x = x.copy() - x.max()
    return np.exp(x) / np.exp(x).sum()


def softmax_regression_SGD(X, y, lr, gamma, n_epoches):  # 带动量SGD求解softmax回归
    np.random.seed(43)
    n, p = X.shape
    k = len(np.unique(y))  # 类别个数
    W, delta = np.random.randn(k, p), 0  # W \in R^{k,p}是全连接层的权重矩阵
    for epoch in range(n_epoches):
        shuffled_idxs, total_loss = np.random.permutation(n), 0
        for idx in shuffled_idxs:
            x, y_real = X[idx, :], y[idx]  # 真随机梯度下降,一次处理一个样本
            y_hat = softmax(W @ x)  # 前向计算：将全连接层的k维输出转为k点分布
            loss = - np.log(y_hat[y_real])  # 交叉熵损失,只对真实标签的预测概率作负对数惩罚
            y_hat[y_real] -= 1  # softmax 层反向传播, 即损失函数(标量)到softmax的k维输入向量的梯度
            grad = np.outer(y_hat, x)  # 链式法则得到损失函数(标量)关于权重矩阵的梯度,形状为kxp = (k x 1) x (1 x p)
            delta = gamma * delta - lr * grad  # gamma 为动量系数，利用当前点的梯度对上次迭代的变化量进行纠正
            W += delta
            total_loss += loss
        if epoch % 20 == 0:
            print(f'epoch {epoch}, avg loss {total_loss / n}')  # 简单起见,输出epoch最后一个批次的损失
    return W


if __name__ == '__main__':
    import sklearn.datasets as sd

    iris = sd.load_iris()
    X = iris.data
    y = iris.target
    W = softmax_regression_SGD(X, y, 1e-3, 0.9, 100)
    y_pred = [softmax(W @ X[i, :]).argmax() for i in range(len(X))]
    print(np.count_nonzero(y_pred == y) / len(y))  # 训练集准确率Accuracy
