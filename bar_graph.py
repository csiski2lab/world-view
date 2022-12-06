# ====================================
# モデルとパラメータを読み込み、
# 棒グラフを表示するプログラム
# ====================================

import matplotlib.pyplot as plt
import numpy as np

def bar_graph(data, label=[]):
    print("[[リアル, アニメ, デフォルメ, 時代]]")
    ret1 = []
    ret2 = []
    ret3 = []
    ret4 = []

    for z in data:
        ret1.append(z[0])
        ret2.append(z[1])
        ret3.append(z[2])
        ret4.append(z[3])
        print(z)

    # グラフ図示
    left = np.arange(len(label))
    width = 0.2
    plt.bar(left+width*0, ret1, width=width, align='center', label="リアル")
    plt.bar(left+width*1, ret2, width=width, align='center', label="アニメ")
    plt.bar(left+width*2, ret3, width=width, align='center', label="デフォルメ")
    plt.bar(left+width*3, ret4, width=width, align='center', label="時代")

    plt.xticks(left+width*1.5, label, fontname="Meiryo")
    
    # 凡例
    plt.legend(loc=1, prop={"family":"Meiryo"}) 

    # 格子線
    gpoint = np.arange(0, 1.1, 0.1)
    plt.hlines(gpoint, -width, len(label)-width, linewidth=0.6, linestyle='dashed', color='gray')

    plt.show()

