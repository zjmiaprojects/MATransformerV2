import time
# from Main.CMT import CMT
# net = CMT()
# from Main.CENet import CE_Net
# net = CE_Net(1,1)
# net_name = 'CE_Net_monu'
# from Main.UTNet.utnet import UTNet
# net = UTNet(1,32)
# net_name = 'utnet_monu'
# from FCT_model import FCT
# net = FCT()
# net_name = 'FCT_monu'
# from Main.swinTransformer.swin import SwinTransformer
# # from Main.swin_nddr2d import SwinTransformer
# model = SwinTransformer()
# from pytorch_dcsaunet.DCSAU_Net import Model
# net=Model(1,1)
# net_name='DCSAU_Net_monu_1e-3'
# from crosslink_net.CrossWnet import UNet
# net = UNet((1,512,512), True)
# net_name = 'CrossWNet_monu'
# from Dual_Swin_Transformer_UNet.DS_TransUNet import UNet
# net=UNet(256, 1,3)
# net_name='DS_TransUNet_monu'
# from Main.resSCNN_sad import resSCNN_sad,DiceLoss
# net = resSCNN_sad()
# model_name = 'SSANet_monu'
from Model import MATransformerV2 as model

net = model()
net_name = 'MATransformerV2_1'
# -- coding: utf-8 --
import torch
import torchvision
from thop import profile

# Model
print('==> Building model..')
# model = torchvision.models.alexnet(pretrained=False)

dummy_input = torch.randn(1, 1, 512, 512)
flops, params = profile(net, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

dataset = [torch.randn(1, 1, 512, 512) for _ in range(100)]

# model = net.to("cuda:0").eval()
net = net.to("cuda:0").eval()
# start = time.time()
time_all = []
all_fps=0
for j in range(30):
    for i in dataset:
        with torch.no_grad():
            cuda_i = i.to("cuda:0")
            start = time.time_ns()
            res = net(cuda_i)
            time_all.append(time.time_ns() - start)
    print(sum(time_all) / len(time_all) / 1e6)
    print(1000 / (sum(time_all) / len(time_all) / 1e6))
    if(j>=10):
        all_fps+=(1000 / (sum(time_all) / len(time_all) / 1e6))
        print("*")
print(all_fps/20)

