cat: 26 cls
dim:  3 regression

tools/
vid2img_kinetics.py  get frames from videos
csv2txt.py           make image dir for every segment of annotation
eval_ers.py          evaluation script
merge_csv.py         merge the result from cat and dim
 
experiments/
tsn_test/default.yaml    config file;when in test, gpu,batchsize->1

x_temporal/
interface/temporal_helper.py         line28 control some return w.r.t. train/test mode
                                     line 423 return the predict results
test.py                              line 46  save the predict results
