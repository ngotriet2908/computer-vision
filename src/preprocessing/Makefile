DEBUG ?= True

make_k_fold:
	python kfold.py --csv_path=/Users/capu/PycharmProjects/ml_mac/data/train/train_set.csv \
					--nfold=5 \
					--output_dir=/Users/capu/PycharmProjects/ml_mac/data/train/kfold \
					--seed=2022

preprocessing_train:
	python preprocessing.py --csv_path=/Users/capu/PycharmProjects/ml_mac/data/train/train_set.csv \
							--data_path=/Users/capu/PycharmProjects/ml_mac/data/train/img \
							--output_dir=/Users/capu/PycharmProjects/ml_mac/data/train/img_png \
							--mode='train'

preprocessing_test:
	python preprocessing.py --csv_path=/Users/capu/PycharmProjects/ml_mac/data/test/test_set.csv \
							--data_path=/Users/capu/PycharmProjects/ml_mac/data/test/img \
							--output_dir=/Users/capu/PycharmProjects/ml_mac/data/test/img_png \
							--mode='test'

preprocessing: preprocessing_train preprocessing_test
all: make_k_fold preprocessing_train preprocessing_test