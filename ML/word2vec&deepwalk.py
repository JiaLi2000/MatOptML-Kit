import networkx as nx
import numpy as np


def generate_pairs(text, window_size):  # corpus为list of str, corpus每个元素表示一个句子，句子每个元素表示一个单词
    vocabulary = list({word for sentence in text for word in sentence.split(' ')})
    word2index = dict(zip(vocabulary, range(len(vocabulary))))  # 存储字典,便于后续由word查找index
    pairs = []
    for sentence in text:
        words_list = sentence.split(' ')
        for i, word in enumerate(words_list):  # 依次以句子中的每个单词为中心词
            for j in range(i - window_size, i + window_size + 1):
                if 0 <= j < len(words_list) and j != i:  # 2w窗口之内且跳过边界及自身
                    pairs.append([word2index[word], word2index[words_list[j]]])  # 把(target,context)pair加入训练集
    return vocabulary, word2index, np.array(pairs)


def index2onehot(index, n):
    onehot = np.zeros(n)
    onehot[index] = 1
    return onehot


def softmax(x):
    x = x.copy() - x.max()
    return np.exp(x) / np.exp(x).sum()


def Skipgram_SGD(pairs, N, lr, gamma, n_epoches):  # 带动量SGD求解softmax回归
    np.random.seed(43)
    n = len(pairs)
    V = pairs.max() + 1  # 类别个数 = 词汇表长度
    W1, W2, delta1, delta2 = np.random.randn(V, N), np.random.randn(N, V), 0, 0  # W1, W2是全连接层的权重矩阵
    for epoch in range(n_epoches):
        shuffled_idxs, total_loss = np.random.permutation(n), 0
        for idx in shuffled_idxs:
            x_index, y_index = pairs[idx]  # 真随机梯度下降,一次处理一个样本
            x = index2onehot(x_index, V)
            # 前向传播
            h = W1.T @ x
            z = W2.T @ h
            a = softmax(z)  # 前向计算：将全连接层的k维输出转为k点分布
            # loss 计算
            loss = - np.log(a[y_index])  # 交叉熵损失,只对真实标签的预测概率作负对数惩罚
            # 反向传播
            a[y_index] -= 1  # softmax 层反向传播, 即损失函数(标量)到softmax的k维输入向量的梯度
            grad_W2 = np.outer(h, a)  # 链式法则得到损失函数(标量)关于权重矩阵的梯度,形状为kxp = (k x 1) x (1 x p)
            grad_W1 = np.outer(x, (W2 @ a))

            delta2 = gamma * delta2 - lr * grad_W2  # gamma 为动量系数，利用当前点的梯度对上次迭代的变化量进行纠正
            delta1 = gamma * delta1 - lr * grad_W1  # gamma 为动量系数，利用当前点的梯度对上次迭代的变化量进行纠正

            W2 += delta2
            W1 += delta1
            total_loss += loss
        if epoch % 10 == 0:
            print(f'epoch {epoch}, avg loss {total_loss / n}')  # 简单起见,输出epoch最后一个批次的损失
    return W1


def generate_random_walks(G, T, L):  # 为G的每个顶点生成T个长为L的随机游走
    np.random.seed(43)
    walks = []
    for node in G.nodes():
        for t in range(T):
            walk = ""
            cur = node
            for i in range(L):
                walk = walk + str(cur) + " "
                if len(list(G.neighbors(cur))) != 0:
                    cur = np.random.choice(list(G.neighbors(cur)))
            walks.append(walk)
    return walks


if __name__ == '__main__':
    text = ["I like pets", "You like dogs", "She likes cats"]
    vocabulary, word2index, pairs = generate_pairs(text, 2)
    Embeddings = Skipgram_SGD(pairs, 10, 1e-3, 0.9, 1000)
    print(Embeddings)

    G = nx.karate_club_graph()
    text = generate_random_walks(G,20,10)
    vocabulary, word2index, pairs = generate_pairs(text, 3)
    Embeddings = Skipgram_SGD(pairs, 20, 1e-3, 0.9, 20)
    print(Embeddings)
