import logging
from math import ceil
import numpy as np

np.set_printoptions(threshold=np.inf)

import torch
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    check_path = '../output/checkpoint/'
    # from Model import MATransformerV2 as model
    #
    # net = model()
    # net_name = 'MATransformerV2_1'
    #
    from unet.unet import UNet
    net7 = UNet(1,1)
    net_name7 = 'unet_data'

    from Dataset.dataset import norm
    save_path = "feature_map_output/data/unet/60_x_code/"
    model_path7 = os.path.join(check_path, net_name7)
    info_path7 = os.path.join(model_path7, 'inform.pkl')
    model_path7 = os.path.join(model_path7, 'val.pkl')
    net7.load_state_dict(torch.load(model_path7, map_location='cpu')['state'])
    #device = torch.device('cuda')
    device = torch.device('cpu')

    net7.to(device)
    net7.eval()
    # ct1 = np.load('../MoNu/test/img/5_1.npy')
    ct1 = np.load('../data/test/image/9.npy')
    ct1_array = norm(ct1)
    ct1_tensor = torch.FloatTensor(ct1_array).to(device)
    ct1_tensor = ct1_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    # output, t1, t2, t3, t = net7(ct1_tensor)
    # visualize_feature_map(t1, save_path, "21_t1", BI,64)
    # visualize_feature_map(t2, save_path, "21_t2", BI,128)
    # visualize_feature_map(t3, save_path, "5_1_t3", BI,256)
    # visualize_feature_map(t, save_path, "60_t4", BI,512)
    outputs,q_loss,t1,t2,t3,x5,x6,x7,x8,x9,codebook_mapping,x_code,x6_up = net7(ct1_tensor)
    x_code=torch.squeeze(x_code).cpu().detach().numpy()
    data_list=[]
    for i in range(0, 1024):
        feature_map_split = x_code[i,:, :]
        feature_map_split = np.expand_dims(feature_map_split,axis=2)
        img_width, img_height, ch = feature_map_split.shape
        feature_map_split = feature_map_split.reshape(1024)
        data_list.append(feature_map_split)
    print(len(data_list))
    print(data_list[0].shape)
    import pandas as pd
    center_list=[]
    center_list.append(data_list[16])
    center_list.append(data_list[55])
    # center_list.append(data_list[60])
    center_list.append(data_list[64])
    center_list.append(data_list[70])
    # center_list.append(data_list[197])

    # class_2=class1+class2+class4
    # class2_list=[]
    # for j in class_2:
    #     class2_list.append(data_list[j])
    k=10
    kms = KMeans(n_clusters=k, max_iter=1000)  # ,n_jobs=2,max_iter=200,init=center_list
    y = kms.fit_predict(data_list)
    # print(y)
    class0 = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    class6 = []
    class7 = []
    class8 = []
    class9 = []
    for k in range(0,len(y)):
        if y[k]==0:
            class0.append(k)
            # class0.append(class_2[k])
        elif y[k]==1:
            class1.append(k)
            # class1.append(class_2[k])
        elif y[k]==2:
            class2.append(k)
            # class2.append(class_2[k])
        elif y[k]==3:
            class3.append(k)
            # class3.append(class_2[k])
        elif y[k]==4:
            class4.append(k)
        elif y[k] == 5:
            class5.append(k)
            # class1.append(class_2[k])
        elif y[k] == 6:
            class6.append(k)
            # class2.append(class_2[k])
        elif y[k] == 7:
            class7.append(k)
            # class3.append(class_2[k])
        elif y[k] == 8:
            class8.append(k)
        else:
            class9.append(k)
    print("class0:",class0)
    print("class1:", class1)
    print("class2:", class2)
    print("class3:", class3)
    print("class4:", class4)
    print("class5:", class5)
    print("class6:", class6)
    print("class7:", class7)
    print("class8:", class8)
    print("class9:", class9)

