import numpy as np
import networkx as nx


def PageRank(G: nx.DiGraph, beta, T):  # 避免泄漏的PageRank，解决含0出度子图、0出度顶点问题
    n, A = G.number_of_nodes(), nx.adjacency_matrix(G).toarray()
    d = A.sum(axis=1)
    a, D_inv = (d == 0).astype(np.int64), np.diag(np.where(d != 0, 1 / d, 0))
    P = D_inv @ A  # 原始候选转移概率矩阵
    P_prime = P + a[None, :] @ np.ones((n, 1))  # 消除0出度点，将转移概率矩阵的非0行替换为全1/n行
    P_tilde = beta * P_prime + (1 - beta) / n * np.ones((n, n))  # 消除0出度子图,转移概率矩阵与全1/n矩阵作权和
    pi = np.full(n, 1 / n)
    for t in range(T):  # 不可约非周期Markov链平稳分布存在且唯一
        pi = P_tilde.T @ pi  # P_tilde主特征值为1,这里相当于用了幂法
    return pi


if __name__ == '__main__':
    edgelist = [(0, 1), (0, 2), (1, 2), (2, 0)]
    G = nx.DiGraph(edgelist)
    pr = nx.pagerank(G, 0.85)
    print(pr)
    pr = PageRank(G, 0.85, 10)
    print(pr)
