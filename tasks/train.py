import fire

from src.classification.operations.train import train_all_fold, train_k_fold


def main(path, check_points_dir, mode='all', folds=5, fold=-1):
    if mode == 'all':
        train_all_fold(path, check_points_dir, folds)
    else:
        train_k_fold(path, check_points_dir, fold)


if __name__ == '__main__':
    fire.Fire(main)
