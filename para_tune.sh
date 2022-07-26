alpha=$1
device=$2


for lr in 2e-4 5e-4
do
	for beta in 0 0.25 0.5 0.75 1 1.25
	do
		python main.py --dataset MIMIC3 --batch_size 50 --alpha $alpha --beta $beta --lr $lr --device $device
	done
done