# ====================================
# データセットを読み込むプログラム
# ====================================

from glob import glob
import numpy as np
from imageio import imread
from skimage.transform import resize
import os

class DataLoader():
    def __init__(self, dataset_path, img_res=(128, 128)):
        self.dataset_path = dataset_path
        self.img_res = img_res
        
    def load_data_multiple_label(self, domain=[], batch_size=1, is_testing=False):
        label = []
        path = []
        for d in domain:
            path.extend(glob(f'{self.dataset_path}/{d}/*.jpg'))
        
        # ランダム get from path
        batch_images = None
        if batch_size == 0:
            batch_images = np.array(path)
        else:
            batch_images = np.random.choice(path, size = batch_size)

        # 画像読み込み　read images
        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)

            # 画像をリサイズ
            img = resize(img, self.img_res)

            if not is_testing:
                if np.random.random() > 0.5:
                    img = np.fliplr(img) # 画像を逆順にする

            imgs.append(img)

            # ディレクトリ名取得
            path = os.path.basename(os.path.dirname(img_path))
            label.append(path)

        imgs = np.array(imgs)/127.5 -1.         # 0~255 -> -1~1
        return imgs.astype("float32"), label    # float64 -> float32

    def load_img(self, path):
        img = self.imread(path)
        img = resize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return imread(path, pilmode='RGB').astype(np.float)
