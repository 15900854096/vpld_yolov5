from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import sys
import glob
import math
import numpy as np
import os 

def score_distr(group,x_lim=(0,1),y_lim=(0,1)):
    '''
    可视化N个类别中每个样本的y分布
    :param group: List[np.ndarray], N类样本标签y组成的数组
    :param x_lim: 横坐标区间
    :param y_lim: 纵坐标区间
    :return:
    '''
    group_num=len(group)
    color_map=["violet","tomato","cyan","salmon","limegreen"]
    fig,axs=plt.subplots(group_num,1)
    dem_labels=[]
    for i in range(group_num):
        axs[i].scatter(group[i],[0.05]*group[i].shape[0],label="class_"+str(i),c=color_map[i])
        # axs[i].xlim(x_lim)
        dem_labels.append("class_"+str(i))
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].yaxis.set_ticks_position('left')
        axs[i].set_xlim(x_lim)
        axs[i].set_ylim(y_lim)
        axs[i].set_yticks([0],labels=['score'])
    fig.legend(dem_labels,loc=(0.45,0.85))
    
    
    
    
catchpath="/home/xuqing/kmenas.cache.npy"
if os.path.exists(catchpath):
    print("catchpath exists!!!")
    lenlist=np.load(catchpath)
else:    
    print("catchpath not exists!!!")
    filelist = []
    p = r"/home/xuqing/ps2.0/labels/"
    filelist += glob.glob(str(p + '/**' + '/*.txt'), recursive=True)
    print(len(filelist))

    lenlist=[]
    for file in filelist:
        with open(file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            for lot in lb:
                tmp = math.sqrt(pow(float(lot[3])-float(lot[1]),2)+ pow(float(lot[4])-float(lot[2]),2))
                lenlist.append(tmp)
    np.save(catchpath, lenlist)

a=np.array(lenlist)
x = a.reshape(-1,1)
myKmeans = KMeans(n_clusters = 2)     # 聚类成3团
myKmeans.fit(x)
centers = list(myKmeans.cluster_centers_)
pred = list(myKmeans.predict(x))
print(centers)
print(pred.count(0),pred.count(1),pred.count(2))

if 0:
    x=list(lenlist)
    y=len(x)*[1]
    plt.scatter(x,y)
    plt.show()




