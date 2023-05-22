import numpy as np


def naive_bayes(X, y, x):
    n, p = X.shape
    k = len(np.unique(y))
    # 训练,即通过先验概率P(Y)及条件概率P(X|Y)学习联合概率分布P(X,Y)
    y_counts = np.array([np.count_nonzero(y == i) for i in np.unique(y)])
    prior_distri = (y_counts + 1) / (n + k)  # 拉普拉斯平滑
    cond_distris = []
    for c in range(k):  # 仅处理离散变量,连续变量请分箱
        cond_distri = []
        for j in range(p):
            col_uni = np.unique(X[:, j])
            s = len(col_uni)
            L = [(np.count_nonzero((X[:, j] == col_uni[l]) & (y == c)) + 1) / (y_counts[c] + s) for l in
                 range(s)]  # 平滑的条件概率P(x_j=第l种取值|y=c)
            cond_distri.append(L)
        cond_distris.append(cond_distri)
    # 预测,即计算不同类别对应后验概率的分子,并选择最大者
    post_probs = []
    for c in range(k):
        prob = prior_distri[c]
        for j in range(p):
            col_uni = np.unique(X[:, j])
            l = np.argwhere(col_uni == x[j])[0][0]  # 不要求特征从0升序
            prob *= cond_distris[c][j][l]
        post_probs.append(prob)
    print(post_probs)
    return np.array(prior_distri), cond_distris, np.array(post_probs).argmax()


if __name__ == '__main__':
    x1 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    x2 = np.array([1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 3])
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    x = np.array([2, 1])
    X = np.stack((x1, x2)).T
    pri, con, la = naive_bayes(X, y, x)
    print(pri)
    print(con)
    print(la)
