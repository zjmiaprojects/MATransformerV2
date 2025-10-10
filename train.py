import os
import torch

from torch.utils.data import DataLoader
from Dataset.MoNudataset import MyDataSet,TestDataSet

from tensorboardX import SummaryWriter

import numpy as np

def dice_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()
    union = output.sum() + target.sum()

    return (2. * intersection + smooth) / (union + smooth)

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

# ******** Hyper Parameters ********
# epoches till stop
EPOCHES = 300
# if there is pretrained parameters
PRETRAIN = False
# epoches trained


#other 2D_model
save_path = 'output/checkpoint/'
if os.path.exists(save_path):
    print('ModuleSavePath : {}  path is existed\n'.format(save_path))
else:
    os.mkdir(save_path)
    print('ModuleSavePath : {}  create successful\n'.format(save_path))

training_set_path_img = 'MoNu/train/img'
training_set_path_lab = r'MoNu/train/lab'
valing_set_path_img = 'MoNu/val/img'
valing_set_path_lab = r'MoNu/val/lab'
# batch size
batch_size = 2
# **********************************
from Model.MATransformerV2 import MATransformerV2 as model

network = model()
# network = unet(1,1)
model_name = 'MATransformerV2_1'

from model import mobile_vit_xx_small
path_xxs = 'mobilevit_xxs.pt'
model_xxs = mobile_vit_xx_small()
from model import mobile_vit_x_small
path_xs = 'mobilevit_xs.pt'
model_xs = mobile_vit_x_small()
from model import mobile_vit_small
path_s = 'mobilevit_s.pt'
model_s = mobile_vit_small()
model_xxs.load_state_dict(torch.load(path_xxs))
model_xs.load_state_dict(torch.load(path_xs))
model_s.load_state_dict(torch.load(path_s))

#layer_3->mobileViTBlock1
modelxxs_layer_3_dict = model_xxs.layer_3[1].state_dict()
mobileViTBlock1_dict = network.vit.mobileViTBlock1[0].state_dict()
print(list(modelxxs_layer_3_dict))
print(list(mobileViTBlock1_dict))
state_dict = {k: v for k, v in modelxxs_layer_3_dict.items() if k in mobileViTBlock1_dict.keys()}
print(list(state_dict))
mobileViTBlock1_dict.update(state_dict)
network.vit.mobileViTBlock1[0].load_state_dict(mobileViTBlock1_dict)

#layer_4->mobileViTBlock2
modelxxs_layer_4_dict = model_xxs.layer_4[1].state_dict()
mobileViTBlock2_dict = network.vit.mobileViTBlock2[0].state_dict()
print(list(modelxxs_layer_4_dict))
print(list(mobileViTBlock2_dict))
state_dict = {k: v for k, v in modelxxs_layer_4_dict.items() if k in mobileViTBlock2_dict.keys()}
print(list(state_dict))
mobileViTBlock2_dict.update(state_dict)
network.vit.mobileViTBlock2[0].load_state_dict(mobileViTBlock2_dict)

#layer_5->mobileViTBlock3
models_layer_4_dict = model_s.layer_4[1].state_dict()
mobileViTBlock3_dict = network.vit.mobileViTBlock3[0].state_dict()
print(list(models_layer_4_dict))
print(list(mobileViTBlock3_dict))
state_dict = {k: v for k, v in models_layer_4_dict.items() if k in mobileViTBlock3_dict.keys()}
print(list(state_dict))
mobileViTBlock3_dict.update(state_dict)
network.vit.mobileViTBlock3[0].load_state_dict(mobileViTBlock3_dict)

check_save_path = os.path.join(save_path, model_name)
if os.path.exists(check_save_path):
    print('ModuleSavePath : {}  path is existed\n'.format(check_save_path))
else:
    os.mkdir(check_save_path)
    print('ModuleSavePath : {}  create successful\n'.format(check_save_path))
if os.listdir(check_save_path) != []:
    model_path = os.path.join(check_save_path, 'val_best.pkl')
    checkpoint = torch.load(model_path)
    network.load_state_dict(checkpoint['state'])
    pre_epoch = checkpoint['epoch']

    best_model_path = os.path.join(check_save_path, 'best_inform.pkl')
    best_model = torch.load(best_model_path)
    best_dice = best_model['dice']
    best_epoch = best_model['epoch']
    print('Continue training')

else:
    pre_epoch = 0
    print('Start training!')
    best_dice = 0
    best_loss = 2.0
    best_epoch = 0


writer_name = 'Result/runs/'
writer = SummaryWriter(writer_name)

# with patch
dataset = MyDataSet(training_set_path_img, training_set_path_lab)
data_loader = DataLoader(dataset, batch_size=batch_size,drop_last=True)

val_dataset = TestDataSet(valing_set_path_img, valing_set_path_lab)
val_loader = DataLoader(val_dataset, batch_size=batch_size,drop_last=True)
print('Data loaded.')

# use GPU if available
device = torch.device("cuda:1")
# device = torch.device("cpu")
print('device:', device)
network.to(device)
# from Utils.diceloss import dice_loss
# criterion = dice_loss()
# from Utils.dice_bce_loss import dice_bce_loss
# criterion = dice_bce_loss()
criterion = torch.nn.BCELoss().to(device)  # other
# optimizer
learning_rate = 1e-3
learning_rate_decay = [500, 750]

# training
for epoch in range(pre_epoch + 1, pre_epoch + 1 + EPOCHES):
    opt = torch.optim.Adam(network.parameters(), lr=learning_rate * (0.1 ** (epoch // 150)), weight_decay=1e-8, )
    #opt = torch.optim.Adam([{'params': weight_p, 'weight_decay': 1e-8}, {'params': bias_p, 'weight_decay': 0}],lr=learning_rate * (0.1 ** (epoch // 150)))
    #start_time = time()
    running_loss = 0.0
    network.train()

    for i, (data1,label) in enumerate(data_loader):
        data1 = data1.to(device)
        # data1 = torch.unsqueeze(data1, dim=1)
        label = torch.unsqueeze(label, dim=1)
        label = label.to(device)
        # outputs = network(data1)
        outputs,vq_loss = network(data1)
        opt.zero_grad()
        # loss = criterion.forward(label, outputs)+vq_loss
        loss = criterion(outputs, label)+vq_loss
        # backwarding
        loss.backward()
        # optimizing
        opt.step()
        # print states info
        running_loss += loss.item() * batch_size

    # print epoch loss and time
    print('[%d, Train loss: %.6f]' % (epoch, running_loss / len(dataset)))
    # print(np.mean(acc_list))
    #print((time() - start_time) // 60, 'minutes per epoch.')

    state = {
        'state': network.state_dict(),
        'epoch': epoch
    }
    # save model_set every epoch every 10 epoch
    # if epoch % 10 == 0 and epoch>=10:
    #     torch.save(state, check_save_path + '/' + str(epoch) + '.pkl')

    writer.add_scalars(model_name, tag_scalar_dict={'train_loss': running_loss / len(dataset)}, global_step=epoch)

    if epoch % 5 == 0:
        val_epoch_loss = 0
        network.eval()
        all_dice = 0
        all_iou = 0
        with torch.no_grad():
            for i, (data1, label) in enumerate(val_loader):
                data1 = data1.to(device)
                # data1 = torch.unsqueeze(data1, dim=1)
                label = torch.unsqueeze(label, dim=1)
                label = label.to(device)
                outputs,vq_loss = network(data1)
                # outputs = network(data1)
                loss = criterion(outputs, label)+vq_loss
                # loss = criterion.forward(label, outputs)+vq_loss
                map = torch.squeeze(outputs).cpu().detach().numpy()
                pred = np.zeros_like(map)
                pred[map >= 0.5] = 1
                lab = torch.squeeze(label).cpu().detach().numpy()
                dice = dice_score(outputs,lab)
                iou = iou_score(outputs,lab)
                all_dice += dice
                all_iou += iou
                # if loss > 1:
                #     print(i)
                val_epoch_loss += loss.item() * batch_size

            val_epoch_loss /= len(val_dataset)
            all_dice /= len(val_dataset)
            all_iou /= len(val_dataset)
            print('%d dice: {%.4f}\n' % (epoch, all_dice))
            print('%d iou: {%.4f}\n' % (epoch, all_iou))
            print('%d Test loss: {%.4f}\n' % (epoch, val_epoch_loss))
            if all_dice > best_dice:
                best_dice = all_dice
                best_epoch = epoch
                torch.save(state, check_save_path + '/' + 'val.pkl')
                best_info={
                    'dice': best_dice,
                    'epoch': best_epoch
                }
                torch.save(best_info, check_save_path + '/' + 'inform.pkl')

            print('*************************************')
            print('since now the best model is epoch {%d}' % (best_epoch))
            print('*************************************')

            writer.add_scalars(model_name, tag_scalar_dict={'val_loss': val_epoch_loss},global_step=epoch)


