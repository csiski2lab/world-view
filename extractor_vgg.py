# ====================================
# VGG16を用いた世界観特徴抽出プログラム
# ====================================

from __future__ import print_function, division
from keras.layers import Dense, Flatten
from keras.layers import BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from data_loader import DataLoader
import numpy as np
import os
import tensorflow as tf
from cv2 import imread, resize, COLOR_BGR2RGB
from yaml import safe_load
from bar_graph import bar_graph
from keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16

class Extractor():
    def __init__(self):
        # 設定ファイル読み込み
        self.read_config()

        # ====================================
        # GPUに切り替え
        print("[GPU]")
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

        # ====================================
        # 入力データの設定        
        self.img_rows = 224     # 行 height 224
        self.img_cols = 224     # 列 width 224
        self.channels = 3       # RGB
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.data_loader = DataLoader(
            dataset_path = self.config['dataset']['train'],
            img_res = (
                self.img_rows,
                self.img_cols
            )
        )
        self.dim = 4
        
        # ====================================
        # モデルの設定                
        vgg = VGG16(input_shape=self.img_shape, weights='imagenet', include_top=False)
        for layer in vgg.layers[:15]:
            layer.trainable = False

        output = Flatten()(vgg.output)
        output = Dense(self.dim)(output)
        output = BatchNormalization()(output)
        output = Activation(activation=self.relu_max(max_value=1.0))(output)
        
        self.model = Model(inputs=vgg.input, outputs = output)
        self.model.summary()        

        self.model.compile(
            loss = 'mse', 
            optimizer= 'adam',
            metrics = ['accuracy']
        )

        # ====================================
        # ラベル読み込み
        f = open(self.config['dataset']['label'], encoding="utf-8")
        self.datasets = safe_load(f)
        f.close()

    # 設定ファイル読み込み
    def read_config(self):
        _path = "config.yaml"
        if not os.path.exists(_path):
            print("[Error] config.yaml not found")
            exit()

        f = open(_path, encoding="utf-8")
        self.config = safe_load(f)
        f.close()

    # ReLU関数の最大値を設定
    def relu_max(self, max_value=1.0):        
        def relu_advanced(x):
            return K.relu(x, max_value=K.cast_to_floatx(max_value))
        return relu_advanced    

    # 正解ラベルの取得
    def get_label(self, label):
        an = []
        for l in label:            
            z = self.datasets[l]
            an.append(z)

        return np.array(an, dtype='float32')

    # 学習
    def train(self, epochs=1, batch_size=1):
        data_size=7168

        # データセットの設定
        img, label = self.data_loader.load_data_multiple_label(
            domain=self.config['dataset']['dir'],
            batch_size=data_size
        )

        os.makedirs("save", exist_ok=True)

        # モデルの保存
        json_string = self.model.to_json()
        open("save/model.json", 'w').write(json_string)

        # チェックポイントの設定
        checkpoint = ModelCheckpoint(
            filepath="save/{epoch:02d}.h5",
            monitor='loss',
            save_best_only=True,
            period=10,
        )

        # 正解ラベル取得
        an = self.get_label(label)

        # 学習
        self.model.fit(img, an, batch_size=batch_size, epochs = epochs, callbacks=[checkpoint])

        # 保存
        self.model.save_weights("save/weight.h5")

if __name__ == '__main__':
    model = Extractor()
    
    # ====================================
    # [学習] 学習時は下記のコメントを解除してください
    # ====================================
    # model.train(epochs=50, batch_size=150)
    # exit()

    # ====================================
    # [テスト]
    # ====================================

    # 重みパラメータ
    model.model.load_weights(model.config['weight'])

    # データセット
    parent = model.config['dataset']['test']
    files = os.listdir(parent)
    
    # ====================================    
    # 棒グラフを表示

    z = []
    fl = []
    for i, f in enumerate(files):
        if not 'jpg' in f:
            continue

        path=f"{parent}\{f}"
        fl.append(f"CG作品{chr(i+65)}")
        in_img = imread(path, COLOR_BGR2RGB)
        in_img = resize(in_img, (model.img_rows, model.img_cols))
        in_img = np.array([in_img])
        in_img = in_img /127.5 - 1.0

        ret = model.model.predict(in_img)
        z.append(ret[0])
        print(f"{f}\t :{ret}")

        i += 1
        if i > 1000:
            break
    z = np.array(z)
    bar_graph(z, fl)

    # ====================================    
    # 結果保存

    text = ""
    for i in z:
        li = [str(v) for v in i]
        text += ",".join(li)
        text += "\n"
    
    # 書き込み
    with open("result.csv", 'w', encoding="utf-8") as f:
        f.write(text)
