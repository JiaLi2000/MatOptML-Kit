import numpy as np
def recall(y_pre, y):  # TPR
    return np.count_nonzero((y == 1) & (y_pre == 1)) / np.count_nonzero(y == 1)
def FPR(y_pre, y):  # 假阳性预测个数/阴性总数，误报率
    return np.count_nonzero((y == 0) & (y_pre == 1)) / np.count_nonzero(y == 0)
def auc_roc(y_score, y):
    y_score_01 = (y_score - y.min()) / (y.max() - y.min())
    pairs = []  # (FPR,TPR/Recall) pairs
    for threshold in np.linspace(0, 1, 10000, endpoint=True):  # 根据阈值离散化,记录(FPR,TPR)对
        y_pre = (y_score_01 >= threshold).astype(int)
        TPR_ = recall(y_pre, y)
        FPR_ = FPR(y_pre, y)
        pairs.append([FPR_, TPR_])
    area = 0
    pre_FPR, pre_TPR = pairs[0]
    for cur_FPR, cur_TPR in pairs[1:]:  # recall随阈值增加而减小，阈值0对应(1,1), 阈值1对应(0,0)
        area += (pre_TPR + cur_TPR) * -(cur_FPR - pre_FPR) / 2  # 梯形法则(小梯形面积求和)代替积分
        pre_FPR, pre_TPR = cur_FPR, cur_TPR  # 下一个小梯形
    return area

