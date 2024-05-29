cfg=$1
ds=$2
python scripts/train.py --config ${cfg} --dataset ${ds} &
python scripts/train_values.py --config ${cfg} --dataset ${ds}