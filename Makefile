DEBUG ?= True

train_fold:
	python tasks/train.py --path=/Users/capu/PycharmProjects/ml_mac/data \
							--check_points_dir=/Users/capu/PycharmProjects/ml_mac/data/checkpoints \
							--mode='fold' \
							--folds=5 \
							--fold=1