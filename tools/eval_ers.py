import os
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import glob
import sys
import numpy as np
import random
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score

num_emotion = {1:'Peace', 2:'Affection', 3:'Esteem', 4:'Anticipation', 5:'Engagement', 6:'Confidence', 7:'Happiness', 8:'Pleasure', 9:'Excitement', 
               10:'Surprise', 11:'Sympathy', 12:'Doubt_Confusion', 13:'Disconnection', 14:'Fatigue', 15:'Embarrassment', 16:'Yearning', 17:'Disapproval', 
               18:'Aversion', 19:'Annoyance',  20:'Anger', 21:'Sensitivity', 22:'Sadness', 23:'Disquietment', 24:'Fear', 25:'Pain', 26:'Suffering'
               }

def eval(gt_file_path, pred_file_path):
    with open(gt_file_path) as f:
        lines_gt = f.readlines()
    with open(pred_file_path) as f:
        lines_pred = f.readlines()

    cat_scores_gt = []
    dim_scores_gt = []
    cat_scores_pred = []
    dim_scores_pred = []
    print('processing data...')
    for i in tqdm(range(len(lines_gt))):
        line_parts_gt = lines_gt[i].strip().split(',')
        line_parts_pred = lines_pred[i].strip().split(',')

        score_gt = line_parts_gt[4:33]
        score_pred = line_parts_pred[4:33]
        # score_pred = line_parts_pred[:29]
        
        cat_score_gt = []
        cat_score_gt_pre = [float(_) for _ in score_gt[:26]]
        for score in cat_score_gt_pre:
            cat_score_gt.append(int(score>0.5))
        if all( _ == 0 for _ in cat_score_gt):  #ignore the data when none of the gt > 0.5
            continue 

        # dim_score_gt = [float(_) for _ in score_gt[26:29]]
        
        cat_score_pred = [float(_) for _ in score_pred[:26]]
        # dim_score_pred = [float(_) for _ in score_pred[26:29]]
        
        dim_score_gt = dim_score_pred = []
         
        cat_scores_gt.append(cat_score_gt)
        dim_scores_gt.append(dim_score_gt)
        cat_scores_pred.append(cat_score_pred)
        dim_scores_pred.append(dim_score_pred)

    cat_scores_gt = np.asarray(cat_scores_gt)
    dim_scores_gt = np.asarray(dim_scores_gt)
    cat_scores_pred = np.asarray(cat_scores_pred)
    dim_scores_pred= np.asarray(dim_scores_pred)
    
    print('calculating ERS...')
    ERS = calculate_ERS(cat_scores_pred, cat_scores_gt, dim_scores_pred, dim_scores_gt)
    print(ERS)
    

def calculate_ERS(cat_pred, cat_true, dim_pred, dim_true):
    values_ap = []
    values_roc = []
    values_r2 = []
    
    cat_pred = cat_pred.T
    cat_true = cat_true.T 
    dim_pred = dim_pred.T
    dim_true = dim_true.T

    for i in range(len(cat_pred)):
        values_ap.append(average_precision_score(cat_true[i], cat_pred[i], average='macro'))
        values_roc.append(roc_auc_score(cat_true[i], cat_pred[i], average='macro'))
    # for i in range(len(dim_pred)):
        # values_r2.append(r2_score(dim_true, dim_pred))

    # ERS = (np.mean(values_r2) + ((np.mean(values_ap) + np.mean(values_roc)) / 2)) / 2    
    ERS =(np.mean(values_ap) + np.mean(values_roc)) / 2
    print('map:', np.mean(values_ap))
    print('mroc:', np.mean(values_roc))

    return ERS

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def vis(gt_file_path, pred_file_path):
    img_save_path = '/data-rbd/hejy/X-Temporal/experiments/test/vis_pred'  
    if os.path.exists(img_save_path):
        os.system('rm -r %s'%img_save_path)
    os.mkdir(img_save_path)

    font_size = 20
    gap_size = 2
    max_text_num = 6
    bg_plate_height = font_size
    bg_plate_width = font_size * max_text_num
    box_plate_dist = 5 
    bg_plate_color = (100, 100, 100)
    text_color = (255, 255, 255)

    with open(gt_file_path) as f:
        lines_gt = f.readlines()
    with open(pred_file_path) as f:
        lines_pred = f.readlines()    
    output = []

    for i, line in enumerate(lines_gt):
        # import ipdb;ipdb.set_trace()
        print('%d/%d'%(i, len(lines_gt)))
        line_parts_gt = line.strip().split(',')
        video_name = os.path.splitext(line_parts_gt[0].split('003/')[-1])[0]
        cat_score_gt = [float(_) for _ in line_parts_gt[4:30]]
        vad_score_gt = line_parts_gt[30:33]

        line_parts_pred = lines_pred[i].strip().split(',')
        cat_score_pred = [float(_) for _ in line_parts_pred[:26]]
        
        if any( sigmoid(_) > 0.5 for _ in cat_score_pred):        
            cat_score_idx_pred = [str(i)  for i in range(len(cat_score_pred)) if sigmoid(cat_score_pred[i])>0.5] 
        else:
            cat_score_idx_pred = [str(cat_score_pred.index(max(cat_score_pred)))]
        
        if any( _ > 0.5 for _ in cat_score_gt):        
            cat_score_idx_gt = [str(i)  for i in range(len(cat_score_gt)) if cat_score_gt[i]>0.5] 
        else:
            cat_score_idx_gt = [str(cat_score_gt.index(max(cat_score_gt)))]
        #str_idx = ''
        #for i in cat_score_idx_pred:
        #    str_idx += ' %d' %i     
        #frame_dir = os.path.join('/data-rbd/hejy/BOLD',os.path.dirname(video_name), os.path.basename(video_name).split('.')[0])
        frame_dir = os.path.join('/data-rbd/hejy/BOLD/frames_test/', video_name)
        assert os.path.exists(frame_dir), 'frame_dir doesnot exists' 
    
        # for i, frame_name in enumerate(glob.glob(frame_dir+'/*')):
        for i, frame_name in enumerate(glob.glob(frame_dir+ '/img_00001.jpg')):
            frame_output_name = os.path.join(img_save_path, 'plot_'+'_'.join(video_name.split('/'))+'.jpg') #+ os.path.basename(frame_name))
            img = cv2.imread(frame_name)
            hh, ww, _ = img.shape
            image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            attr_vis_cnt = 0
            for cat_idx in cat_score_idx_pred:
                cat_idx = int(cat_idx)
                if cat_idx:
                    vis_str = num_emotion[cat_idx]
                    draw = ImageDraw.Draw(image_pil)
                    bg_plate_x1 = box_plate_dist
                    bg_plate_y1 = (bg_plate_height + gap_size * 2) * attr_vis_cnt
                    bg_plate_x2 = bg_plate_x1 + bg_plate_width
                    bg_plate_y2 = bg_plate_y1 + bg_plate_height
                    draw.rectangle((bg_plate_x1, bg_plate_y1, bg_plate_x2, bg_plate_y2), fill=bg_plate_color)
                    font = ImageFont.truetype('/data-rbd/hejy/Scripts/FangZhengFangSongJianTi-1.ttf', font_size, encoding='utf-8')
                    # font = ImageFont.load_default()
    
                    text_plate_x1 = bg_plate_x1+3
                    text_plate_y1 = bg_plate_y1
                    # draw.text((text_plate_x1, text_plate_y1), vis_str, text_color)
                    draw.text((text_plate_x1, text_plate_y1), vis_str, text_color, font)
                    attr_vis_cnt += 1
            attr_vis_cnt_gt = 0
            for cat_idx in cat_score_idx_gt:
                cat_idx = int(cat_idx)
                if cat_idx:
                    vis_str = num_emotion[cat_idx]
                    draw = ImageDraw.Draw(image_pil)
                    bg_plate_x1 = ww - bg_plate_width - box_plate_dist
                    bg_plate_y1 = (bg_plate_height + gap_size * 2) * attr_vis_cnt_gt
                    bg_plate_x2 = bg_plate_x1 + bg_plate_width
                    bg_plate_y2 = bg_plate_y1 + bg_plate_height
                    draw.rectangle((bg_plate_x1, bg_plate_y1, bg_plate_x2, bg_plate_y2), fill=bg_plate_color)
                    font = ImageFont.truetype('/data-rbd/hejy/Scripts/FangZhengFangSongJianTi-1.ttf', font_size, encoding='utf-8')
                    # font = ImageFont.load_default()
    
                    text_plate_x1 = bg_plate_x1+3
                    text_plate_y1 = bg_plate_y1
                    # draw.text((text_plate_x1, text_plate_y1), vis_str, text_color)
                    draw.text((text_plate_x1, text_plate_y1), vis_str, (255,0,0), font)
                    attr_vis_cnt_gt += 1
            img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_output_name, img)

if __name__ == "__main__":
    # csv_file_path = "/data-rbd/hejy/BOLD/BOLD_public/annotations/train.csv"
    gt_file_path = "/data-rbd/hejy/BOLD/BOLD_public/annotations/val.csv"
    pred_file_path = "/data-rbd/hejy/X-Temporal/tools/pred.csv"
    eval(gt_file_path, pred_file_path)
    # vis(gt_file_path, pred_file_path)
