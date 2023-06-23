import numpy as np
import networkx as nx


def LPA(G: nx.Graph, T=10):
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
    G = nx.karate_club_graph()
    label = LPA(G)
    print(label)
    pos = nx.spring_layout(G)  # 节点的布局为spring型
    plt.figure(figsize=(8, 6))  # 图片大小
    nx.draw_networkx(G, pos=pos, node_color=label)
    plt.show()