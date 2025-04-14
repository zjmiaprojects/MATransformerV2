import torch.nn as nn
import torch
import os
from imageio import imread
import numpy as np
from sklearn import metrics

from hausdorff import hausdorff_distance
# from distance import surface_distance as surfdist
import cv2
import matplotlib.pyplot as plt
import math


def get_contours(img):
    """获取连通域

    :param img: 输入图片
    :return: 最大连通域
    """
    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0]

def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- gt
    y_pred -- 测试集目标预测值

    返回:
    mae -- MAE 评价指标
    """

    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae


def calRVD(binary_GT,binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    RVD_s, RVD_t = 0,0
    if np.max(binary_GT)!=0:
        for i in range(row):
            for j in range(col):
                if binary_GT[i][j] == 1:
                    RVD_s += 1
                if binary_R[i][j]  == 1:
                    RVD_t += 1
        RVD = RVD_t/RVD_s - 1
    elif np.max(binary_GT)==0 and np.max(binary_R)==0:
        RVD=1
    return RVD


def Hd(gt, pred):

    hd = hausdorff_distance(gt, pred, distance='euclidean')
    # surface_distances = surfdist.compute_surface_distances(
    #     gt, pred, spacing_mm=(1.0, 1.0, 1.0))
    # hd = surfdist.compute_robust_hausdorff(surface_distances, 95)

    return hd*0.95

import numpy as np
# from medpy.metric.binary import __surface_distances


def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float = 3, spacing: tuple = None, connectivity=1):
    """
    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!
    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image
    This implementation natively supports 2D and 3D images. Whether other dimensions are supported depends on the
    __surface_distances implementation in medpy
    :param a: image 1, must have the same shape as b
    :param b: image 2, must have the same shape as a
    :param threshold: distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!
    (if spacing = (1, 1(, 1)) then one voxel=1mm so the threshold is effectively in voxels)
    must be a tuple of len dimension(a)
    :param spacing: how many mm is one voxel in reality? Can be left at None, we then assume an isotropic spacing of 1mm
    :param connectivity: see scipy.ndimage.generate_binary_structure for more information. I suggest you leave that
    one alone
    :return:
    """
    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if np.max(b) != 0 and np.max(a) != 0:
        if spacing is None:
            spacing = tuple([1 for _ in range(len(a.shape))])
        a_to_b = __surface_distances(a, b, spacing, connectivity)
        b_to_a = __surface_distances(b, a, spacing, connectivity)

        numel_a = len(a_to_b)
        numel_b = len(b_to_a)

        tp_a = np.sum(a_to_b <= threshold) / numel_a
        tp_b = np.sum(b_to_a <= threshold) / numel_b

        fp = np.sum(a_to_b > threshold) / numel_a
        fn = np.sum(b_to_a > threshold) / numel_b

        dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0
    elif np.max(a) == 0 and np.max(b) == 0:
        dc = 1
    else:
        dc = 0
    return dc


def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt      = surface_distances["surfel_areas_gt"]
    surfel_areas_pred    = surface_distances["surfel_areas_pred"]
    overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    surface_dice = (overlap_gt + overlap_pred) / (
      np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
    return surface_dice


def calDSI(binary_GT,binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s,DSI_t = 0,0
    if np.max(binary_GT) != 0:
        for i in range(row):
            for j in range(col):
                if binary_GT[i][j] == 1 and binary_R[i][j] == 1:
                    DSI_s += 1
                if binary_GT[i][j] == 1:
                    DSI_t += 1
                if binary_R[i][j]  == 1:
                    DSI_t += 1
        DSI = 2*DSI_s/(DSI_t+1e-8)
    elif np.max(binary_GT) == 0 and np.max(binary_R) == 0:
        DSI = 1
    else:
        DSI = 0
    # print(DSI)
    return DSI


if __name__ == "__main__":


    
    label_path = "./data/test/mask"
    pred_path = "./data/outing/data/test/img"
    files = os.listdir(pred_path)

    files.sort()

    loss_mae = 0
    loss_hd95 = 0
    loss_nsd = 0
    loss_dsc = 0

    np_mae = []
    np_hd95 = []
    np_nsd = []
    np_dsc = []

    i = 0
    for idx in range(len(files)):
        if files[idx].endswith('.nii'):
            # name = files[idx].split('-')[-1]
            # label_p = os.path.join(label_path, 'train-labels-' + name)
            # pred_p = os.path.join(pred_path, 'train-volume-' + name)

            label_p = os.path.join(label_path, files[idx])
            pred_p = os.path.join(pred_path, files[idx])

            label = imread(label_p)
            pred = imread(pred_p)

            pred_new = np.zeros([pred.shape[0], pred.shape[1]])
            gt_new = np.zeros([pred.shape[0], pred.shape[1]])

            pred_new[:, :] = ((pred == 1) * 1)
            gt_new[:, :] = ((label == 1) * 1)
            # print(np.max(gt_new))

            # """
            mae = metrics.mean_absolute_error(gt_new, pred_new)
            loss_mae += mae
            np_mae.append(mae)
          
            hd95 = Hd(gt_new, pred_new)
            loss_hd95 += hd95
            np_hd95.append(hd95)
            
            # nsd = normalized_surface_dice(gt_new, pred_new)
            # loss_nsd += nsd
            # np_nsd.append(nsd)
# """
            dsc = calDSI(gt_new, pred_new)
            loss_dsc += dsc
            np_dsc.append(dsc)

            i += 1

            print(i, files[idx])
            print("MAE:", mae)
            # print("RVD:", rvd, "   RVD_1:", rvd_1)
            print("HD95:", hd95)
            # print("NSD:", nsd)
            print("DSC:", dsc)
            print("-" * 50)

    print(i)
    print("DSC:", '%.4f' % np.mean(np_dsc), '%.4f' % np.std(np_dsc, ddof=1))
    print("HD95:", '%.4f' % np.mean(np_hd95), '%.4f' % np.std(np_hd95, ddof=1))
    print("MAE:", '%.4f' % np.mean(np_mae), '%.4f' % np.std(np_mae, ddof=1))
    # print("RVD:", loss_rvd/i, "   RVD_1:", math.sqrt(loss_rvd_1/i))
    # print("NSD:", '%.4f' % np.mean(np_nsd), '%.4f' % np.std(np_nsd, ddof=1))












