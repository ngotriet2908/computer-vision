import os.path

import cv2
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm


def convert_png(csv_path, data_path, output_dir, mode='train'):
    csv_df = pd.read_csv(csv_path, index_col="Id")
    img_paths = [os.path.join(data_path, f'{mode}_{idx}.npy') for idx, _ in csv_df.iterrows()]
    print("The train set contains {} examples.".format(len(csv_df)))

    for i, img_path in enumerate(tqdm(img_paths)):
        image = np.load(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_dir, f'{mode}_{i}.png'), image)


if __name__ == '__main__':
    fire.Fire(convert_png)
