import os
import pandas as pd
import numpy as np
import fire
from sklearn.model_selection import KFold


def make_kfold_csv(csv_path, nfold, output_dir, seed):
    csv = pd.read_csv(csv_path)
    kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(csv)):
        csv1 = csv.iloc[np.concatenate([train_idx, val_idx])]
        csv1.to_csv(os.path.join(output_dir, f"kfold_{fold + 1}.csv"), index=False)


if __name__ == '__main__':
    fire.Fire(make_kfold_csv)
