from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import os
np.set_printoptions(threshold=np.inf)

# Kmeans聚类
def Kmeans(iter,k, data):
    # cluster_center = data[rd.sample(list(range(len(data))), k)]
    # cluster_center = [data[0],data[2],data[12],data[29],data[40],data[48],data[54],data[156]]
    cluster_center = [data[0],data[2], data[5],data[6],data[10],data[11]]
    # cluster_center = [data[1], data[7]]
    # print("list(range(len(data))):",list(range(len(data))))
    # print("cluster_center:",cluster_center)
    Cluster_tag = np.zeros((len(data),))
    dist = np.zeros((k,))

    while iter > 0:
        print(iter,"_iter")
        n = 0
        for i, sample in enumerate(data):
            # print("i:",i)
            # print("sample:",sample)
            for center in range(k):
                # print("center:",center)
                # print("cluster_center[center]:", cluster_center[center])
                _distance = distance(sample, cluster_center[center])
                dist[center]=_distance
            # print(dist)
            # print("_distance:",_distance)
            # 得到该序列最小值的索引
            max_idx = np.argmax(dist)
            # print("max_idx",max_idx)
            if Cluster_tag[i] != max_idx:
                Cluster_tag[i] = max_idx
            else:
                n+=1
        print(n)
        for tag in range(k):
            class_list=[]
            # print("data[0].type:",type(data[0]))
            for c in range(len(Cluster_tag)):
                if Cluster_tag[c]==tag:
                    class_list.append(data[c])
            centroid = np.mean(class_list, axis=0)
            cluster_center[tag] = centroid
            for j in range(len(cluster_center)):
                plt.axis("off")
                plt.imshow(cluster_center[j].astype('uint8'))  # , cmap=plt.cm.jet
                plt.savefig('D:/pythonProject/unet/Utils/feature_map_output/data/MATransformerV2/kMeans_9_x_code/PSNR_6class/' + str(j) + "_iter{}.jpg".format(100-iter), bbox_inches='tight', pad_inches=-0.1)

        if n==len(Cluster_tag):
            print("iter:",iter)
            for j in range(len(cluster_center)):
                print(cluster_center[j].shape)
                # cluster_center[j] = np.expand_dims(cluster_center[j], axis=2)
                plt.axis("off")
                plt.imshow(cluster_center[j].astype('uint8'))  # , cmap=plt.cm.jet
                plt.savefig('D:/pythonProject/unet/Utils/feature_map_output/data/MATransformerV2/kMeans_9_x_code/PSNR_6class/' + str(j) + "_final.jpg", bbox_inches='tight', pad_inches=-0.1)
            break
        iter = iter - 1
    # print('质心坐标：\n', cluster_center)

    # j = 0
    # while j < len(range(k)):
    #     plt.scatter(cluster_center[j][0], cluster_center[j][1], c='red', marker='*', s=50)
    #     j = j + 1
    return Cluster_tag


# 导入数据
def Import_data():
    # import_data = np.array(pd.read_csv(r'C:\Users\86159\Desktop\flame.txt', sep='\t', header=None, usecols=[0, 1]))
    # print(import_data)
    # import_data = np.array(8,int)
    img_list = []

    path='D:/pythonProject/unet/Utils/feature_map_output/data/MATransformerV2/x_code_9/'
    for i in range(1024):
        img=Image.open(path+str(i)+'_9_x_code.jpg')
        img = np.array(img)
        img_list.append(img)

    return img_list


# 计算欧式距离
def distance(x, y):
    # print(x)
    # print(y)
    # return ssim(x, y, multichannel=True)
    return psnr(x, y)
    # return np.sum((x - y) ** 2, axis=1) ** 0.5


# 聚类结果可视化展示
def show(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.6, s=20)
    plt.grid(axis='both', linewidth=0.2)
    plt.show()


# 导出聚类结果
def Clustering_result(m, n):
    i = -1
    while i < n.shape[0] - 1:
        i = i + 1
        with open(r'D:\soft\PyCharm 2022.1.3\pythonproject\kms\answer.txt', 'a+', encoding='utf-8') as f:
            print(m[i][0], m[i][1], int(n[i]), sep='\t', file=f)


if __name__ == '__main__':

    data = Import_data()
    Cluster_tag = (Kmeans(100,6, data))
    # dict1 = dict(Counter(Cluster_tag))
    print('聚类标签：\n', Cluster_tag)#, '\n每类中的数据个数:\n', dict1
    class0 = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    class6 = []
    class7 = []
    for k in range(0, len(Cluster_tag)):
        if Cluster_tag[k] == 0:
            class0.append(k)
            # class0.append(class_0[k])
        elif Cluster_tag[k] == 1:
            class1.append(k)
            # class1.append(class_0[k])
        elif Cluster_tag[k] == 2:
            class2.append(k)
            # class2.append(class_2[k])
        elif Cluster_tag[k] == 3:
            class3.append(k)
        elif Cluster_tag[k] == 4:
            class4.append(k)
        elif Cluster_tag[k] == 5:
            class5.append(k)
        # elif Cluster_tag[k] == 6:
        #     class6.append(k)
        # elif Cluster_tag[k] == 7:
        #     class7.append(k)
    print("class0:", class0)
    print("class1:", class1)
    print("class2:", class2)
    print("class3:", class3)
    print("class4:", class4)
    print("class5:", class5)
    # print("class6:", class6)
    # print("class7:", class7)



    #
    # print('0:', list(Cluster_tag).count(0), '1:', list(Cluster_tag).count(1))
    #
    # Clustering_result(data, Cluster_tag)
    # ans = pd.read_csv(r'D:\soft\PyCharm 2022.1.3\pythonproject\kms\answer.txt', sep='\t', header=None)
    # print('聚类结果展示(聚类结果已保存至answer.txt):\n', ans)
    #
    # show(data, Cluster_tag)
