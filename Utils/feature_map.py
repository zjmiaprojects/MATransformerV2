import logging
from math import ceil

import torch

import numpy as np
import matplotlib.pyplot as plt
import os
def visualize_feature_map(img_batch,out_path,type,BI,chans):
    print("img_batch.shape:",img_batch.shape)
    feature_map = torch.squeeze(img_batch)
    feature_map = feature_map.detach().cpu().numpy()
    print("feature_map.shape:", feature_map.shape)
    feature_map_sum = feature_map[0,:, :]
    print("feature_map_sum.shape:", feature_map_sum.shape)
    feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
    print("feature_map_sum.shape:", feature_map_sum.shape)
    for i in range(0, chans):
        feature_map_split = feature_map[i,:, :]
        feature_map_split = np.expand_dims(feature_map_split,axis=2)
        if i > 0:
            feature_map_sum +=feature_map_split
        #print(feature_map_split)
        #feature_map_split = BI.transform(feature_map_split)
        print(i)
        plt.axis("off")
        plt.imshow(feature_map_split)#, cmap=plt.cm.jet
        plt.savefig(out_path + str(i) + "_{}.jpg".format(type),bbox_inches='tight',pad_inches = -0.1)
        # plt.xticks()
        # plt.yticks()
        # plt.axis('off')

    # feature_map_sum = BI.transform(feature_map_sum)
    # plt.imshow(feature_map_sum)
    # plt.savefig(out_path + "sum_{}.jpg".format(type))
    # print("save sum_{}.jpg".format(type))

class BilinearInterpolation(object):
    def __init__(self, w_rate: float, h_rate: float, *, align='center'):
        if align not in ['center', 'left']:
            logging.exception(f'{align} is not a valid align parameter')
            align = 'center'
        self.align = align
        self.w_rate = w_rate
        self.h_rate = h_rate

    def set_rate(self,w_rate: float, h_rate: float):
        self.w_rate = w_rate    # w 的缩放率
        self.h_rate = h_rate    # h 的缩放率

    # 由变换后的像素坐标得到原图像的坐标    针对高
    def get_src_h(self, dst_i,source_h,goal_h) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_i = float(dst_i * (source_h/goal_h))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_i = float((dst_i + 0.5) * (source_h/goal_h) - 0.5)
        src_i += 0.001
        src_i = max(0.0, src_i)
        src_i = min(float(source_h - 1), src_i)
        return src_i
    # 由变换后的像素坐标得到原图像的坐标    针对宽
    def get_src_w(self, dst_j,source_w,goal_w) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_j = float(dst_j * (source_w/goal_w))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_j = float((dst_j + 0.5) * (source_w/goal_w) - 0.5)
        src_j += 0.001
        src_j = max(0.0, src_j)
        src_j = min((source_w - 1), src_j)
        return src_j

    def transform(self, img):
        source_h, source_w, source_c = img.shape  # (235, 234, 3)
        goal_h, goal_w = round(
            source_h * self.h_rate), round(source_w * self.w_rate)
        new_img = np.zeros((goal_h, goal_w, source_c), dtype=np.uint8)

        for i in range(new_img.shape[0]):       # h
            src_i = self.get_src_h(i,source_h,goal_h)
            for j in range(new_img.shape[1]):
                src_j = self.get_src_w(j,source_w,goal_w)
                i2 = ceil(src_i)
                i1 = int(src_i)
                j2 = ceil(src_j)
                j1 = int(src_j)
                x2_x = j2 - src_j
                x_x1 = src_j - j1
                y2_y = i2 - src_i
                y_y1 = src_i - i1
                new_img[i, j] = img[i1, j1]*x2_x*y2_y + img[i1, j2] * \
                    x_x1*y2_y + img[i2, j1]*x2_x*y_y1 + img[i2, j2]*x_x1*y_y1
        return new_img


if __name__ == '__main__':
    check_path = '../output/checkpoint/'
    from Model import MATransformerV2 as model
    net = model()
    net_name = 'MATransformerV2_1'

    from Dataset.dataset import norm
    save_path = "feature_map_output/data/x_60/"
    model_path = os.path.join(check_path, net_name)
    info_path = os.path.join(model_path, 'inform.pkl')
    model_path = os.path.join(model_path, 'val.pkl')
    net.load_state_dict(torch.load(model_path, map_location='cpu')['state'])
    #
    device = torch.device('cuda')

    net.to(device)
    net.eval()
    BI = BilinearInterpolation(8, 8)
    # ct1 = np.load('../MoNu/test/img/5_1.npy')
    ct1 = np.load('../data/test/image/92.npy')
    ct1_array = norm(ct1)
    ct1_tensor = torch.FloatTensor(ct1_array).to(device)
    ct1_tensor = ct1_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    # output, t1, t2, t3, t = net7(ct1_tensor)
    # visualize_feature_map(t1, save_path, "21_t1", BI,64)
    # visualize_feature_map(t2, save_path, "21_t2", BI,128)
    # visualize_feature_map(t3, save_path, "5_1_t3", BI,256)
    # visualize_feature_map(t, save_path, "60_t4", BI,512)
    outputs,q_loss,t1,t2,t3,x5,x6,x7,x8,x9,codebook_mapping,x_code,x6_up = net(ct1_tensor)
    # visualize_feature_map(t1, save_path, "60_t1", BI,64)
    # visualize_feature_map(t2, save_path, "60_t2", BI,128)
    # visualize_feature_map(t3, save_path, "60_t3", BI,256)
    # visualize_feature_map(x5, "x_center_92", "92_x_center", BI, 1024)
    visualize_feature_map(x_code, "x_code_92", "92_x_code", BI,1024)
    # visualize_feature_map(x6_up, save_path, "60_x6_up", BI, 512)
    # visualize_feature_map(x6, save_path, "60_x6", BI, 512)
    # visualize_feature_map(x7, save_path, "60_x7", BI, 256)
    # visualize_feature_map(x8, save_path, "60_x8", BI, 128)
    # visualize_feature_map(x9, save_path, "60_x9", BI, 64)
    # ct2 = np.load('../data/test/image/9.npy')
    # ct2_array = norm(ct2)
    # ct2_tensor = torch.FloatTensor(ct2_array).to(device)
    # ct2_tensor = ct2_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

    #output, _, t1, t2, t3, t = net7(ct2_tensor)
    # visualize_feature_map(t1, save_path, "9_t1", BI,64)
    # visualize_feature_map(t2, save_path, "9_t2", BI,128)
    # visualize_feature_map(t3, save_path, "9_t3", BI,256)
    #visualize_feature_map(t, save_path, "9_t4", BI,1024)