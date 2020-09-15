import os
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import glob
import sys
import numpy as np
import random
# reload(sys)
# sys.setdefaultencoding('utf-8')

from sklearn.metrics import average_precision_score, roc_auc_score, r2_score




num_emotion = {1:'Peace', 2:'Affection', 3:'Esteem', 4:'Anticipation', 5:'Engagement', 6:'Confidence', 7:'Happiness', 8:'Pleasure', 9:'Excitement', 
               10:'Surprise', 11:'Sympathy', 12:'Doubt_Confusion', 13:'Disconnection', 14:'Fatigue', 15:'Embarrassment', 16:'Yearning', 17:'Disapproval', 
               18:'Aversion', 19:'Annoyance',  20:'Anger', 21:'Sensitivity', 22:'Sadness', 23:'Disquietment', 24:'Fear', 25:'Pain', 26:'Suffering'
               }


def csv2txt(csv_file_path, txt_file_path):
    with open(csv_file_path) as f:
        lines = f.readlines()
        random.shuffle(lines)
    output = []
    for line in tqdm(lines):
        line_parts = line.strip().split(',')
        video_name = os.path.splitext(line_parts[0].split('003/')[-1])[0]
        cat_score = [float(_) for _ in line_parts[4:30]]
        vad_score = line_parts[30:33]
        
        if any( _ > 0.5 for _ in cat_score):        
            cat_score_idx = [str(i)  for i in range(len(cat_score)) if cat_score[i]>0.5] 
        else:
            cat_score_idx = [str(cat_score.index(max(cat_score)))]
        #str_idx = ''
        #for i in cat_score_idx:
        #    str_idx += ' %d' %i     
        #frame_dir = os.path.join('/data-rbd/hejy/BOLD',os.path.dirname(video_name), os.path.basename(video_name).split('.')[0])
        frame_dir = os.path.join('/data-rbd/hejy/BOLD/frames_test/', video_name)
        assert os.path.exists(frame_dir), 'frame_dir doesnot exists'
        frame_num = len(os.listdir(frame_dir))
        output.append('%s %d %s' %(video_name, frame_num, ','.join(cat_score_idx + vad_score)))
        # output.append('%s %d %s' %(video_name, frame_num, ','.join(cat_score_idx)))
        var_info = [cat_score_idx, frame_dir,video_name]
        #vis(var_info)

    with open(txt_file_path, 'w') as f:
        f.write('\n'.join(output))   

def csv2output_eval(csv_file_path):
    with open(csv_file_path) as f:
        lines = f.readlines()
        # random.shuffle(lines)

    cat_score_gts = []
    cat_scores = []
    dim_score_gts = []
    for line in tqdm(lines):
        line_parts = line.strip().split(',')
        video_name = os.path.splitext(line_parts[0].split('003/')[-1])[0]
        cat_score = [float(_) for _ in line_parts[4:30]]
        dim_score = [float(_) for _ in line_parts[30:33]]
        cat_score_gt = []
        for score in cat_score:
            cat_score_gt.append(int(score>0.5))
        if all( _ == 0 for _ in cat_score_gt):
            continue
        cat_score_gts.append(cat_score_gt)
        cat_scores.append(cat_score)
        dim_score_gts.append(dim_score)
    dim_scores = dim_score_gts
    # return cat_scores, cat_score_gts, dim_scores, dim_score_gts
    cat_scores = np.asarray(cat_scores)
    cat_score_gts = np.asarray(cat_score_gts)
    dim_scores = np.asarray(dim_scores)
    dim_score_gts = np.asarray(dim_score_gts)
    ERS = calculate_ERS(cat_scores, cat_score_gts, dim_scores, dim_score_gts)
    print(ERS)

def calculate_ERS(cat_pred, cat_true, dim_pred, dim_true):
    # y_pred = y_pred.detach().cpu().numpy()
    # y_true = y_true.detach().cpu().numpy()
    values_ap = []
    values_roc = []
    import ipdb;ipdb.set_trace()   
    for i in range(len(cat_pred)):
        values_ap.append(average_precision_score(cat_true[i], cat_pred[i], average='macro'))
        # print(i)
        values_roc.append(roc_auc_score(cat_true[i], cat_pred[i], average='macro'))
        
    import ipdb;ipdb.set_trace()
    value_r2 = r2_score(dim_true, dim_pred)
    ERS = (value_r2 + ((np.mean(values_ap) + np.mean(values_roc)) / 2)) / 2    
    return ERS


def vis(var_info):
    cat_score_idx = var_info[0]
    frame_dir = var_info[1]
    video_name = var_info[2]

    img_save_path = '/data-rbd/hejy/X-Temporal/experiments/test/vis'  
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path) 

    font_size = 20
    gap_size = 2
    max_text_num = 6
    bg_plate_height = font_size
    bg_plate_width = font_size * max_text_num
    box_plate_dist = 5 
    bg_plate_color = (100, 100, 100)
    text_color = (255, 255, 255)
    
    # for i, frame_name in enumerate(glob.glob(frame_dir+'/*')):
    for i, frame_name in enumerate(glob.glob(frame_dir+ '/img_00001.jpg')):
        frame_output_name = os.path.join(img_save_path, 'plot_'+'_'.join(video_name.split('/'))+'.jpg') #+ os.path.basename(frame_name))
        img = cv2.imread(frame_name)
        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        attr_vis_cnt = 0
        for cat_idx in cat_score_idx:
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
        img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_output_name, img)



if __name__ == "__main__":
    csv_file_path = "/data-rbd/hejy/BOLD/BOLD_public/annotations/train.csv"
    txt_file_path =  "train_cat_dim.txt"
    # csv_file_path = "/data-rbd/hejy/BOLD/BOLD_public/annotations/val.csv"
    # txt_file_path =  "val_cat_dim.txt"
    # csv2output_eval(csv_file_path)
    csv2txt(csv_file_path, txt_file_path)

