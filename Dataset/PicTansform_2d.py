import numpy as np
import random

#old should be a opened pic
def filp_LeftRight(old):
    if len(old.shape) > 2:
        old = old.transpose(1, 2, 0)
        new = np.flip(old, axis=0)
        new = new.transpose(2, 0, 1)
    else:
        new = np.flip(old, axis=0)

    return new
def rotation1(old):
    if len(old.shape) > 2:
        old = old.transpose(1, 2, 0)
        new = np.rot90(old, 1)
        new = new.transpose(2, 0, 1)
    else:
        new = np.rot90(old, 1)

    return new
def rotation2(old):
    if len(old.shape) > 2:
        old = old.transpose(1, 2, 0)
        new = np.rot90(old, -1)
        new = new.transpose(2, 0, 1)
    else:
        new = np.rot90(old, -1)

    return new
def rotation3(old):
    if len(old.shape) > 2:
        old = old.transpose(1, 2, 0)
        new = np.rot90(old, 2)
        new = new.transpose(2, 0, 1)
    else:
        new = np.rot90(old, 2)

    return new
def rotation4(old):
    if len(old.shape) > 2:
        old = old.transpose(1, 2, 0)
        new = np.rot90(old, 3)
        new = new.transpose(2, 0, 1)
    else:
        new = np.rot90(old, 3)

    return new

def filp_UpDown(old):
    if len(old.shape)>2:
        old = old.transpose(1, 2, 0)
        new = np.flip(old, axis=1)
        new = new.transpose(2, 0, 1)
    else:
        new = np.flip(old, axis=1)

    return new

def crop(image,label,size=(1,1,1)):
    shape = image.shape
    range_z = shape[0]-size[0]
    range_x = shape[1] - size[1]
    range_y = shape[2] - size[2]
    ran_z = random.randint(0,range_z)
    ran_x = random.randint(0,range_x)
    ran_y = random.randint(0,range_y)
    img = image[ran_z:ran_z+size[0],ran_x:ran_x+size[1],ran_y:ran_y+size[2]]
    lab = label[ran_z:ran_z+size[0],ran_x:ran_x+size[1],ran_y:ran_y+size[2]]
    return img,lab

