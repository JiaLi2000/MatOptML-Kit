import numpy as np
import networkx as nx
import scipy.sparse as sp

def louvain(G:nx.Graph, T):
    label = np.array(G.nodes)  # 各点自成一簇，节点编号作为簇编号
    V = np.array(G.nodes).copy()
    is_steady = True
    for t in range(T):
        # np.random.shuffle(V)
        for i in range(5): # 第一阶段
            for v in V:
                neighbor = np.array(list(G.neighbors(v)))
                gains = np.array([gain(v,u) for u in neighbor])
                max_u = gains.argmax()
                if gains[max_u] > 0:
                    label[neighbor[max_u]] = label[v]

    return 0

def gain(u,v):


    return 0


def get_ncut(G: nx.graph, phi: np.array) -> float:
    n, W = len(phi), nx.adjacency_matrix(G, sorted(G.nodes))
    k = phi.max() + 1
    Y_0 = sp.coo_matrix(([1] * n, (range(n), phi)), dtype=int).tocsr()
    D = W.dot(Y_0).toarray()  # d_ij = sum_v of W_{iv}, where v is in cluster j
    degrees = D.sum(axis=1)
    volumes = np.array([degrees[phi == j].sum() for j in range(k)])
    cuts = volumes - (D * Y_0.toarray()).sum(axis=0)  # the right item just is association of clusters.
    f = (cuts / volumes).sum()
    return f



def modularity(G:nx.Graph, label): #无向带权图模块度计算
    n, W = len(label), nx.adjacency_matrix(G, sorted(G.nodes))
    k = label.max() + 1
    Y = sp.coo_matrix(([1] * n, (range(n), label)), dtype=int).tocsr()
    D = W.dot(Y).toarray()  # d_ij = sum_v of W_{iv}, where v is in cluster j
    degrees = D.sum(axis=1)
    m = degrees.sum()/2
    volumes = np.array([degrees[label == j].sum() for j in range(k)])
    asso = (D * Y.toarray()).sum(axis=0)  # the right item just is association of clusters.
    Q = (asso/(2*m) - (volumes/(2*m))**2).sum()
    return Q


def LPA(G, T=10):
    np.random.seed(43)
    label = np.array(G.nodes)  # 各点自成一簇，节点编号作为簇编号
    V = np.array(G.nodes).copy()
    for t in range(T):
        np.random.shuffle(V)
        for v in V:
            neighbor = np.array(list(G.neighbors(v)))
            label[v] = np.bincount(label[neighbor]).argmax()  # 将每个顶点的标签设置为其邻居中标签的众数
    _, relabeled_label = np.unique(label, return_inverse=True)  # 将节点编号表示的簇编号重编号，使之从0升序
    return relabeled_label

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def label2com(label):
        k = label.max() + 1
        return [list(np.argwhere(label == i).flatten()) for i in range(k)]

    G = nx.karate_club_graph()
    label = LPA(G,5)
    print(label)
    coms = label2com(label)
    print(coms)
    print(nx.community.modularity(G,coms ))
    print(modularity(G,label))











