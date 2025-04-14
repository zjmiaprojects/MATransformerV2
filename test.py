import re
from Dataset.dataset import norm
from Dataset.dataset import MyDataSet,TestDataSet
import os

import numpy as np
import torch
import collections
from time import time

mae = []
dice_intersection = 0.0
dice_union = 0.0
file_name = []
threhold = 0.5



def dice_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()
    union = output.sum() + target.sum()

    return (2. * intersection + smooth) / (union + smooth)

def mae_value(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae

def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # output_ = output.astype('uint8')
    # target_ = target[target > 0.5]
    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)

# def Recall(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     intersection = (output * target).sum()
#     return (intersection + smooth) / (target.sum() + smooth)
#
# def Precision(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#
#     intersection = (output * target).sum()
#
#     return (intersection + smooth) / (output.sum() + smooth)
def F1_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    Recall = (intersection + smooth) / (target.sum() + smooth)
    Precision = (intersection + smooth) / (output.sum() + smooth)
    return (2 * Recall * Precision  + smooth) / (Recall + Precision + smooth)

def FNR_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()
    FN = target.sum() - intersection
    union = target.sum()

    return (FN + smooth) / (union + smooth)

def UR_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    FN = target.sum() - intersection
    FP = output.sum() - intersection
    return (FN + smooth) / (FN + FP + intersection + smooth)

check_path = 'output/checkpoint/'
training_set_path_img = 'MoNu/test/img/'
training_set_path_lab = r'MoNu/test/lab/'
valing_set_path_img = 'MoNu/test/img/'
valing_set_path_lab = r'MoNu/test/lab/'



pred_path = '/home/zbz/DataSet/LITS/test1/output/'
out_path = '/home/smh/Downloads/data/yywdata/data/out'
excel_path = '/home/zbz/DataSet/LITS/test1/excel/'
time_all = []
dice_list=[]
iou_list = []
F1_list = []
UR_list = []
FNR_list = []

from Model import MATransformerV2 as model

net = model()
# network = unet(1,1)
net_name = 'MATransformerV2_1'

device = torch.device('cuda:0')
# net.to(device)
net.to(device)

model_path = os.path.join(check_path, net_name)
info_path = os.path.join(model_path, 'best_inform.pkl')
model_path = os.path.join(model_path, 'val_best.pkl')


all_fps=0
for j in range(21):
    net.load_state_dict(torch.load(model_path,map_location='cpu')['state'])
    net.eval()

    data_list = os.listdir(valing_set_path_img)

    for i in range(len(data_list)):

        label = data_list[i]
        ct1 = np.load(os.path.join(valing_set_path_img, data_list[i]))
        ct1_array = norm(ct1)
        with torch.no_grad():
            ct1_tensor = torch.FloatTensor(ct1_array).to(device)
            ct1_tensor = ct1_tensor.unsqueeze(dim=0)
            start = time()
            # outputs,in_dice = net(ct1_tensor)
            # outputs,_ = net(ct1_tensor)
            outputs = net(ct1_tensor)
            # logits, v_featuremap, h_featuremap = net(ct1_tensor)
            # outputs = torch.sigmoid(logits)
            end = time()
            time_all.append(end-start)
            probability_map = torch.squeeze(outputs).cpu().detach().numpy()

        pred_seg = np.zeros_like(probability_map)
        pred_seg[probability_map >= (threhold)] = 1
        seg_array = np.load(os.path.join(valing_set_path_lab, label))
        seg_array[seg_array > 0] = 1

        dice = dice_score(pred_seg,seg_array)
        dice_list.append(dice)
        mae1 = mae_value(seg_array,pred_seg)
        mae.append(mae1)
        iou = iou_score(pred_seg, seg_array)
        iou_list.append(iou)
        F1 = F1_score(pred_seg, seg_array)
        F1_list.append(F1)
        UR = UR_score(pred_seg, seg_array)
        UR_list.append(UR)
        FNR = FNR_score(pred_seg, seg_array)
        FNR_list.append(FNR)

        # print : dice global
    print('dice',np.mean(dice_list))
    print('F1',np.mean(F1_list))
    print('FNR',np.mean(FNR_list))
    print('iou',np.mean(iou_list))
    print('UR',np.mean(UR_list))
    print('mae',np.mean(mae))

    fps = 1 / np.mean(time_all)
    print('average time:', np.mean(time_all) / 1)
    print('average fps:',fps)
    print('fastest time:', min(time_all) / 1)
    print('fastest fps:',1 / min(time_all))
    print('slowest time:', max(time_all) / 1)
    print('slowest fps:',1 / max(time_all))
    print("################################################")
    if(j>=1):
        all_fps+=fps
        print(j)

print("fps:",all_fps/20)








