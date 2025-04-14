import os
import Dataset.PicTansform_2d as pt
import random
import matplotlib.pyplot as plt
import re
import torch
from torch.autograd import Variable
from torch.utils import data
import numpy as np
from torchvision.transforms import *

upper, lower = 200, -200

class MyDataSet(data.Dataset):

    def __init__(self, data_dir, label_dir):
        self.data_list = os.listdir(data_dir)
        self.data_dir = data_dir
        self.label_dir = label_dir



    def __getitem__(self, index):
        data1 = self.data_list[index]
        label = data1.split('.')[0]+'_Corrected.npy'

        data1 = os.path.join(self.data_dir, data1)
        label = os.path.join(self.label_dir, label)

        data1 = np.load(data1)
        label = np.load(label)

        # data1[data1 > upper] = upper
        # data1[data1 < lower] = lower
        # data2[data2 > upper] = upper
        # data2[data2 < lower] = lower
        # ind = random.randint(0, 7)
        # data1, label = choose_tans(data1, label, ind)


        data1 = norm(data1/1.0)
        label = label/1.0
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
        # label = norm(label/1.0)


        data1 = torch.FloatTensor(data1)
        label = torch.FloatTensor(label)

        return data1, label


    def __len__(self):
        return len(self.data_list)


class TestDataSet(data.Dataset):
    def __init__(self,testdata_dir, testlabel_dir):
        self.testdata_list = os.listdir(testdata_dir)
        self.testdata_dir = testdata_dir
        self.testlabel_dir = testlabel_dir


    def __getitem__(self, index):
        data1 = self.testdata_list[index]
        label = self.testdata_list[index].split('.')[0]+'_Corrected.npy'
        data1 = os.path.join(self.testdata_dir, data1)
        label = os.path.join(self.testlabel_dir, label)

        data1 = norm(np.load(data1))
        label = np.load(label) / 1.0
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
        # label = norm(np.load(label) / 1.0)


        data1 = torch.FloatTensor(data1)
        label = torch.FloatTensor(label)

        return data1, label

    def __len__(self):
        return len(self.testdata_list)


def norm(data):
    max = np.max(data)
    min = np.min(data)
    normal_data = (data - min) /((max - min) + 1e-6)
    return normal_data

def choose_tans(image,label,index):
    if index == 0:
        img = pt.filp_LeftRight(image)
        lab = pt.filp_LeftRight(label)
    elif index == 1:
        img = pt.rotation1(image)
        lab = pt.rotation1(label)
    elif index == 2:
        img = pt.filp_UpDown(image)
        lab = pt.filp_UpDown(label)
    elif index == 3:
        img = pt.rotation2(image)
        lab = pt.rotation2(label)
    elif index == 4:
        img = pt.rotation3(image)
        lab = pt.rotation3(label)
    elif index == 5:
        img = pt.rotation4(image)
        lab = pt.rotation4(label)
    else:
        img =image
        lab =label
    return img,lab


