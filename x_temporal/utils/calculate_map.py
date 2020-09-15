import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score

def calculate_mAP(y_pred, y_true):
    # y_pred = y_pred[0][0].detach().cpu().numpy()
    # y_true = y_true[0][:26].detach().cpu().numpy()
    # values = []
    # for i in range(len(y_pred)): 
    #     # y_pred[i] = sigmoid(y_pred[i])
    #     # # print(y_pred[i])
    #     # sort_idx = np.argsort(y_pred[i])[::-1]
    #     # y_pred[i]= np.zeros_like(y_pred[i])
    #     # # print(y_pred[i])
    #     # for idx in sort_idx[:3]:
    #     #     y_pred[i][idx] = 1
    #     # # print(y_pred[i])
        
    #     values.append(average_precision_score(y_true[i], sigmoid(y_pred[i]), average='macro'))
    #     # values.append(average_precision_score(y_true[i], y_pred[i], average='macro'))
    #     # values.append(roc_auc_score(y_true[i], y_pred[i], average='macro'))
    #     # import ipdb;ipdb.set_trace()
    #     # r2 = r2_score(y_true, y_pred)
    # return r2

    cat_pred = y_pred[0].detach().cpu().numpy()
    cat_true = y_true[:,:26].detach().cpu().numpy()
    dim_pred = y_pred[1].detach().cpu().numpy()
    dim_true = y_true[:,26:29].detach().cpu().numpy()
    values_ap = []
    values_roc = []
    values_r2 = []
    
    # cat_pred = cat_pred.T
    # cat_true = cat_true.T 
    # dim_pred = dim_pred.T
    # dim_true = dim_true.T

    for i in range(len(cat_pred)):
        # import ipdb;ipdb.set_trace()
        values_ap.append(average_precision_score(cat_true[i], cat_pred[i], average='macro'))
        values_roc.append(roc_auc_score(cat_true[i], cat_pred[i], average='macro'))
    for i in range(len(dim_pred)):
        values_r2.append(r2_score(dim_true, dim_pred))

    ERS = (np.mean(values_r2) + ((np.mean(values_ap) + np.mean(values_roc)) / 2)) / 2    
    # ERS =(np.mean(values_ap) + np.mean(values_roc)) / 2
    # print('map:', np.mean(values_ap))
    # print('mroc:', np.mean(values_roc))

    # return np.mean(values) * 100
    return ERS

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
