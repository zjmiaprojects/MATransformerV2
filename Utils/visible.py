import torch
import numpy as np
from Dataset.dataset import norm
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
#check_path = '/home/smh/Downloads/SLWNet/output/checkpoint/'
check_path = '../output/checkpoint/'
# valing_set_path_img = '../data/test/image'
# valing_set_path_lab = r'../data/test/label'
from Dataset.dataset import norm

from Model import MATransformerV2 as model

net = model()
net_name = 'MATransformerV2_1'
device = torch.device('cuda:0')
net.to(device)

model_path = os.path.join(check_path, net_name)
info_path = os.path.join(model_path, 'inform.pkl')
model_path = os.path.join(model_path, 'val.pkl')
net.load_state_dict(torch.load(model_path, map_location='cpu')['state'])
# data_list = os.listdir(valing_set_path_img)
# #os.path.join(valing_set_path_img, data_list[41])
# ct1 = np.load('../data/test/image/60.npy')
# ct1 = np.load('../MoNu/test/img/8_1.npy')
# ct1 = np.load('/home/zyt/code/SLWNet_1/em_data/test/img_512/22_3.npy')
ct1 = np.load('/export/home/xh/zyt/em/test/img_512/103_1.npy')
ct1_array = norm(ct1)
ct1_tensor = torch.FloatTensor(ct1_array).to(device)
ct1_tensor = ct1_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
# outputs,_ = net(ct1_tensor)
outputs = net(ct1_tensor)
outputs = torch.sigmoid(outputs)
probability_map = torch.squeeze(outputs).cpu().detach().numpy()
pred_seg = np.zeros_like(probability_map)
pred_seg[probability_map >= (0.5)] = 1
sci = pred_seg*255
a = Image.fromarray(sci).convert('L')
# # a.save('MVnet2_7_codebook_condconv_60'+'.jpg')
# a.save('ssa_22_3'+'.jpg')#22_3
a.save('103_1'+'.jpg')

plt.imshow(a, cmap='gray')

# data_list = os.listdir(valing_set_path_img)
# print(data_list)
# all_fps=0
# for i in range(len(data_list)):

    # print(data_list[i])
    # label = data_list[i]
    #
    # ct1 = np.load(os.path.join(valing_set_path_img, data_list[i]))
    # img = ct1
    # ig = Image.fromarray(img).convert("L")
    #
    # ct1_array = norm(ct1)
    # #ct1 = np.load('../data/test/image/60.npy')
    # ct1_tensor = torch.FloatTensor(ct1_array).to(device)
    # ct1_tensor = ct1_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    #
    # # outputs = net(ct1_tensor)
    # outputs2 = net2(ct1_tensor)
    # # outputs3 = net3(ct1_tensor)
    # logits, v_featuremap, h_featuremap = net4(ct1_tensor)
    # outputs4 = torch.sigmoid(logits)
    # outputs5 = net5(ct1_tensor)
    # outputs6 = net6(ct1_tensor)
    # outputs7,_ = net7(ct1_tensor)
    # outputs8,_ = net8(ct1_tensor)
    # # probability_map = torch.squeeze(outputs).cpu().detach().numpy()
    # # pred_seg = np.zeros_like(probability_map)
    # # pred_seg[probability_map >= (0.5)] = 1
    # # sci = pred_seg*255
    # # a = Image.fromarray(sci).convert('L')
    #
    # probability_map2 = torch.squeeze(outputs2).cpu().detach().numpy()
    # pred_seg2 = np.zeros_like(probability_map2)
    # pred_seg2[probability_map2 >= (0.5)] = 1
    # sci2 = pred_seg2 * 255
    # b = Image.fromarray(sci2).convert('L')
    #
    # # probability_map3 = torch.squeeze(outputs3).cpu().detach().numpy()
    # # pred_seg3 = np.zeros_like(probability_map3)
    # # pred_seg3[probability_map3 >= (0.5)] = 1
    # # sci3 = pred_seg3 * 255
    # # c = Image.fromarray(sci3).convert('L')
    #
    # probability_map4 = torch.squeeze(outputs4).cpu().detach().numpy()
    # pred_seg4 = np.zeros_like(probability_map4)
    # pred_seg4[probability_map4 >= (0.5)] = 1
    # sci4 = pred_seg4 * 255
    # d = Image.fromarray(sci4).convert('L')
    #
    # probability_map5 = torch.squeeze(outputs5).cpu().detach().numpy()
    # pred_seg5 = np.zeros_like(probability_map5)
    # pred_seg5[probability_map5 >= (0.5)] = 1
    # sci5 = pred_seg5 * 255
    # e = Image.fromarray(sci5).convert('L')
    #
    # probability_map6 = torch.squeeze(outputs6).cpu().detach().numpy()
    # pred_seg6 = np.zeros_like(probability_map6)
    # pred_seg6[probability_map6 >= (0.5)] = 1
    # sci6 = pred_seg6 * 255
    # f = Image.fromarray(sci6).convert('L')
    #
    # lab = np.load(os.path.join(valing_set_path_lab, label))
    # g = lab * 255
    # g = Image.fromarray(g).convert('L')
    # g.save('gt_em_' + data_list[i] + '.jpg')
    #
    # probability_map7 = torch.squeeze(outputs7).cpu().detach().numpy()
    # pred_seg7 = np.zeros_like(probability_map7)
    # pred_seg7[probability_map7 >= (0.5)] = 1
    # sci7 = pred_seg7 * 255
    # h = Image.fromarray(sci7).convert('L')
    #
    # probability_map8 = torch.squeeze(outputs8).cpu().detach().numpy()
    # pred_seg8 = np.zeros_like(probability_map8)
    # pred_seg8[probability_map8 >= (0.5)] = 1
    # sci8 = pred_seg8 * 255
    # l = Image.fromarray(sci8).convert('L')
    #
    # # plt.subplot(241)
    # # plt.axis("off")
    # # a.save('unet_monu_'+data_list[i]+'.jpg')
    # # plt.imshow(a,cmap='gray')
    #
    # ax1=plt.subplot(241)
    # plt.axis("off")
    # ax1.set_title('cmt')
    # b.save('cmt_em_' + data_list[i] + '.jpg')
    # plt.imshow(b,cmap='gray',label='cmt')
    # # plt.subplot(243)
    # # plt.axis("off")
    # # c.save('cenet_monu_' + data_list[i] + '.jpg')
    # # plt.imshow(c, cmap='gray')
    #
    # ax2=plt.subplot(242)
    # plt.axis("off")
    # ax2.set_title('crossW')
    # d.save('crossnet_em_' + data_list[i] + '.jpg')
    # plt.imshow(d, cmap='gray',label='crossW')
    #
    # ax3=plt.subplot(243)
    # plt.axis("off")
    # ax3.set_title('DCSAU')
    # e.save('dcsau_em_' + data_list[i] + '.jpg')
    # plt.imshow(e, cmap='gray',label='DCSAU')
    #
    # ax5=plt.subplot(245)
    # plt.axis("off")
    # ax5.set_title('MA-trans')
    # f.save('mvnet2_em_' + data_list[i] + '.jpg')
    # plt.imshow(f, cmap='gray')
    #
    # ax6=plt.subplot(246)
    # plt.axis("off")
    # ax6.set_title('ours')
    # h.save('mvnet2_7_em_' + data_list[i] + '.jpg')
    # plt.imshow(h, cmap='gray',label='ours')
    #
    # ax4=plt.subplot(244)
    # plt.axis("off")
    # ax4.set_title('ssa')
    # l.save('ssa_em_' + data_list[i] + '.jpg')
    # plt.imshow(l, cmap='gray',label='ssa')
    #
    # ax7=plt.subplot(247)
    # plt.axis("off")
    # ax7.set_title('img')
    # ig.save('img_em_' + data_list[i] + '.jpg')
    # plt.imshow(ig, cmap='gray',label='img')
    #
    # ax8=plt.subplot(248)
    # plt.axis("off")
    # ax8.set_title('gt')
    # # g.save('gt_em_' + data_list[i] + '.jpg')
    # plt.imshow(g, cmap='gray',label='gt')
    # plt.savefig(data_list[i]+'.jpg')
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
    # plt.axis("off")
    # plt.title(data_list[i])
    # plt.show()







