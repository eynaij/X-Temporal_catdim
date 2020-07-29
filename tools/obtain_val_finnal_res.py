import pandas as pd
import numpy as np
import csv

def calculate(temp):
	temp_new = []
	for i in range(26):
		if len(temp) > 2:
			max_num = max(temp[:,i])
			min_num = min(temp[:,i])
			temp_new.append((sum(temp[:,i]) - max_num - min_num)/(len(temp)-2))
		else:
			temp_new.append(sum(temp[:,i])/(len(temp)))

	return temp_new


data = pd.read_csv('./se_densenet_val/efficientnet-b7_val_sig.csv',header=None)

info = dict()

for i in range(len(data)):
	t_data = list(data.iloc[i])
	video = t_data[0]
	person_id = t_data[2]
	if video not in list(info.keys()):
		info[video] = dict()
	if person_id not in list(info[video].keys()):
		info[video][person_id] = []
	info[video][person_id].append(list(t_data[3:]))


val_data = pd.read_csv('./val.csv',header=None)

res = []
count = 0
for i in range(len(val_data)):
	t_val = list(val_data.iloc[i])
	video = t_val[0]
	video = 'crop-val'+video[video.find('/'):-4]
	person_id = t_val[1]

	print(i, ' : ', video)

	if video in list(info.keys()) and person_id in list(info[video].keys()):
		t_info = info[video][person_id]
		t_info = np.array(t_info)
		print(np.shape(t_info))
		print()
		
		t_info = calculate(t_info)
		
	else:
		t_info = [0]*26
		count = count + 1
	res.append(t_info)

res = np.array(res)
res = pd.DataFrame(res)
res.to_csv('./se_densenet_val/efficient-b7_val_26_res.csv')

print('-------------------------------')
print('count = ', count)
print()


	
