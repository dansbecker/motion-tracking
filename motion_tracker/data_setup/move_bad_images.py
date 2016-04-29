import cv2
import numpy as np
import os
import shutil
import sys
from os.path import join
from tqdm import tqdm

if __name__ == "__main__":
    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    os.makedirs(dest_dir, exist_ok=True)

    fnames = os.listdir(source_dir)
    for fname in tqdm(fnames):
        img_path = join(source_dir, fname)
        try:
            img = cv2.imread(img_path)
            assert isinstance(img, np.ndarray)
        except:
            dest_path = join(dest_dir, fname)
            shutil.move(img_path, dest_path)

    n_files_in_dest = len(os.listdir(dest_dir))
    print('Files in destination: ', n_files_in_dest)
