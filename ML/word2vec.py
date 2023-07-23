import numpy as np

settings = {'window_size': 2,
            'n': 3,
            'epochs': 500,
            'learning_rate': 0.01}


def generate_dataset(self, corpus):
    word_counts = dict()
    for row in corpus:
        for word in row.split(' '):
            if word_counts.get(word, -1) == -1:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
    V = list(word_counts.keys())
    distinct_words_count = len(V)
    self.word_index = dict((word, i) for i, word in enumerate(self.words_list))  # {单词:索引}
    self.index_word = dict((i, word) for i, word in enumerate(self.words_list))  # {索引:单词}

    training_data = []
    for row in corpus:
        tmp_list = row.split(' ')  # 语句单词列表
        sent_len = len(tmp_list)  # 语句长度
        for i, word in enumerate(tmp_list):  # 依次访问语句中的词语
            w_target = self.word2onehot(tmp_list[i])  # 中心词ont-hot表示
            w_context = []  # 上下文
            for j in range(i - self.window, i + self.window + 1):
                if j != i and j <= sent_len - 1 and j >= 0:
                    w_context.append(self.word2onehot(tmp_list[j]))
            training_data.append([w_target, w_context])  # 对应了一个训练样本

    return training_data


class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

    def word2onehot(self, word):
        """
        :param word: 单词
        :return: ont-hot
        """
        word_vec = [0 for i in range(0, self.v_count)]  # 生成v_count维度的全0向量
        word_index = self.word_index[word]  # 获得word所对应的索引
        word_vec[word_index] = 1  # 对应位置位1
        return word_vec

    def train(self, training_data):
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))  # 随机生成参数矩阵
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
        for i in range(self.epochs):
            self.loss = 0

            for data in training_data:
                w_t, w_c = data[0], data[1]  # w_t是中心词的one-hot，w_c是window范围内所要预测此的one-hot
                y_pred, h, u = self.forward_pass(w_t)

                train_loss = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)  # 每个预测词都是一对训练数据，相加处理
                self.back_prop(train_loss, h, w_t)

                for word in w_c:
                    self.loss += - np.dot(word, np.log(y_pred))

            print('Epoch:', i, "Loss:", self.loss)

    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_pred = self.softmax(u)
        return y_pred, h, u

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # 防止上溢和下溢。减去这个数的计算结果不变
        return e_x / e_x.sum(axis=0)

    def back_prop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.dot(self.w2, e.T).reshape(-1)
        self.w1[x.index(1)] = self.w1[x.index(1)] - (self.lr * dl_dw1)  # x.index(1)获取x向量中value=1的索引，只需要更新该索引对应的行即可
        self.w2 = self.w2 - (self.lr * dl_dw2)


if __name__ == '__main__':
    corpus = ['natural language processing and machine learning is fun and exciting']
    w2v = word2vec()
    training_data = w2v.generate_training_data(corpus)
    w2v.train(training_data)
