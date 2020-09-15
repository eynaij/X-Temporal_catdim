T=`date +%m%d%H%M`
ROOT=../..
cfg=default.yaml

export PYTHONPATH=$ROOT:$PYTHONPATH

# python $ROOT/x_temporal/train.py --config $cfg | tee log.train_tin-mit-8-resnet101 
# nohup python -u $ROOT/x_temporal/train.py --config $cfg > log.train_tin-mit-8-resnet101 & #.$T
# nohup python -u $ROOT/x_temporal/train.py --config $cfg > log.train_tin-kinetics-8-resnet101 & #.$T
nohup python -u $ROOT/x_temporal/train.py --config $cfg > log.train_tsn_kinetics_resnet50_catdim.$T & #
# python -u $ROOT/x_temporal/train.py --config $cfg



