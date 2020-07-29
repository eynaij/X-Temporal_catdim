# coding=utf-8
# author=yphacker

import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from conf import config
from model.net import Net
from utils.data_utils import MyDataset
from utils.data_utils import test_transform
import torch.nn as nn
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model):
    #test_path_list = ['{}/{}.jpg'.format(config.image_test_path, x) for x in range(0, data_len)]

    #test_path_list = ['{}/{}'.format(config.image_test_path, x) for x in os.listdir(config.image_test_path)]

    csv_path = '/data/beijing/dataset/BOLD/BOLD_public/val.csv'
    info = pd.read_csv(csv_path)
    test = list(info['filename'])
 
    test_path_list = []
    for i in range(len(test)):
        if os.path.exists(config.image_test_path + '/' + test[i]):
            test_path_list.append(config.image_test_path + '/' + test[i])
    #print(test_path_list)
    #exit()

    test_data = np.array(test_path_list)

    test_dataset = MyDataset(test_data, test_transform, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    

    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            batch_x = batch_x.to(device)
            # compute output
            probs = model(batch_x)
            # preds = torch.argmax(probs, dim=1)
            # pred_list += [p.item() for p in preds]
            pred_list.extend(probs.cpu().numpy())
     
            #print('----------------------------------------------------------------------')
            #print('batch_x = ', batch_x)
            #print('probs = ', probs)
            #print(np.shape(batch_x))
            #print(np.shape(probs))
            #print()
            #exit()
   
    return pred_list,test_data


def multi_model_predict():
    preds_dict = dict()
    for model_name in model_name_list:
        model = Net(model_name).to(device)
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
        print()
        print('model path is : ',model_save_path)
        #model_save_path = './save_model/model_1/se_densenet121.bin'

        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_save_path))

        pred_list, test_data = predict(model)
        test_data = list(test_data)

        submission = pd.DataFrame(pred_list)
        # submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
        submission.to_csv('{}/{}_val.csv'
                              .format(config.submission_path, model_name), index=False, header=False)
        #submission.to_csv('./submission_test/se_densenet121/submission_sedensenet121_val.csv')
        preds_dict['{}'.format(model_name)] = pred_list
    pred_list = get_pred_list(preds_dict)
    
    #print()
    #print(preds_dict.keys())
    #a = preds_dict['se_densenet121']
    #print(np.shape(a))
    #print()
    #print(pred_list)
    #print(np.shape(pred_list))
    #exit()

    #submission = pd.DataFrame({"id": test_data, "label": pred_list})
    test_data = pd.DataFrame(test_data)
    pred_list = pd.DataFrame(pred_list)
    submission = pd.concat([test_data,pred_list],axis=1)
    #submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
    submission.to_csv('submission_val.csv', index=False, header=False)


def file2submission():
    preds_dict = dict()
    for model_name in model_name_list:
        for fold_idx in range(5):
            df = pd.read_csv('{}/{}_fold{}_submission.csv'
                             .format(config.submission_path, model_name, fold_idx), header=None)
            preds_dict['{}_{}'.format(model_name, fold_idx)] = df.values
    pred_list = get_pred_list(preds_dict)
    
    submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
    submission.to_csv('submission.csv', index=False, header=False)


def get_pred_list(preds_dict):
    pred_list = []
    for i in range(data_len):
        prob = None
        for model_name in model_name_list:
            if prob is None:
                prob = preds_dict['{}'.format(model_name)][i] * ratio_dict[model_name]
            else:
                prob += preds_dict['{}'.format(model_name)][i] * ratio_dict[model_name]
        pred_list.append(np.argmax(prob))

    return pred_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=120, type=int, help="batch size")
    parser.add_argument("-m", "--model_names", default='efficientnet-b7', type=str, help="model select")
    parser.add_argument("-type", "--pred_type", default='model', type=str, help="model select")
    parser.add_argument("-mode", "--mode", default=2, type=int, help="1:加权融合，2:投票融合")
    parser.add_argument("-r", "--ratios", default='1', type=str, help="融合比例")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    model_name_list = args.model_names.split('+')
    ratio_dict = dict()
    ratios = args.ratios
    ratio_list = args.ratios.split(',')
    for i, ratio in enumerate(ratio_list):
        ratio_dict[model_name_list[i]] = int(ratio)
    mode = args.mode
    data_len = len(os.listdir(config.image_test_path))
    if args.pred_type == 'model':
        multi_model_predict()
    elif args.pred_type == 'file':
        file2submission()
