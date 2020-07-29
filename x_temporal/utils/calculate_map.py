import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score

def calculate_mAP(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    values = []
    for i in range(len(y_pred)): 
        # y_pred[i] = sigmoid(y_pred[i])
        # # print(y_pred[i])
        # sort_idx = np.argsort(y_pred[i])[::-1]
        # y_pred[i]= np.zeros_like(y_pred[i])
        # # print(y_pred[i])
        # for idx in sort_idx[:3]:
        #     y_pred[i][idx] = 1
        # # print(y_pred[i])
        
        # values.append(average_precision_score(y_true[i], sigmoid(y_pred[i]), average='macro'))
        values.append(average_precision_score(y_true[i], y_pred[i], average='macro'))
        # values.append(roc_auc_score(y_true[i], y_pred[i], average='macro'))
        # import ipdb;ipdb.set_trace()
        # r2_score(y_true, y_pred)
        
    return np.mean(values) * 100

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
