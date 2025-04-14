import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from matplotlib import pyplot as plt



# Í¼Æ¬Ô¤´¦Àí
def img_preprocess():
    valing_set_path_img = r'/home/smh/Downloads/data/yywdata/data/test/image/60.npy'
    valing_set_path_lab = r'/home/smh/Downloads/data/yywdata/data/test/label/60.npy'
    img = np.load(valing_set_path_img)
    coff = img
    ct1_tensor = torch.FloatTensor(img)
    img = ct1_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

    lab = np.load(valing_set_path_lab)
    ct_tensor = torch.FloatTensor(lab)
    lab = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

    return img,lab,coff


# ¶¨Òå»ñÈ¡ÌÝ¶ÈµÄº¯Êý
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# ¶¨Òå»ñÈ¡ÌØÕ÷Í¼µÄº¯Êý
def farward_hook(module, input, output):
    fmap_block.append(output)


# ¼ÆËãgrad-cam²¢¿ÉÊÓ»¯
def cam_show_img(img, feature_map, grads, out_dir):
    H, W = img.shape
    image = []
    image.append(img)
    image.append(img)
    image.append(img)
    img = np.array(image).transpose((1,2,0))

    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    cam_img = 0.5 * heatmap + 0.5 * img * 200


    path_cam_img = os.path.join(out_dir, "unet22.jpg")
    cv2.imwrite(path_cam_img, cam_img)


if __name__ == '__main__':

    output_dir = '/home/smh/Downloads/data/out_pic/hotpic/'
    criterion = torch.nn.BCELoss()


    fmap_block = list()
    grad_block = list()


    img_input,label,img = img_preprocess()

    # ¼ÓÔØ squeezenet1_1 Ô¤ÑµÁ·Ä£ÐÍ
    from Main.MVnet3 import MVNet
    net = MVNet()

    net.load_state_dict(torch.load(r'/home/smh/Downloads/SLWNet/output/checkpoint/mvnet_invT_covid/175.pkl')['state'])
    # pthfile = './squeezenet1_1-f364aa15.pth'
    # net.load_state_dict(torch.load(pthfile))
    net.eval()  # 8

    # ×¢²áhook
    # set_trace()
    # net.features[-1].expand3x3.register_forward_hook(farward_hook)	# 9
    # net.features[-1].expand3x3.register_backward_hook(backward_hook)

    net.vit.center_conv.register_forward_hook(farward_hook)  # 9
    net.vit.center_conv.register_backward_hook(backward_hook)

    # forward
    output = net(img_input)

    net.zero_grad()
    loss = criterion(output, label)
    loss.backward()

    # Éú³Écam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # ±£´æcamÍ¼Æ¬
    cam_show_img(img, fmap, grads_val, output_dir)
