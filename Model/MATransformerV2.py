from functools import partial
import torch
import torch.nn as nn
from MobileVITBlock import MobileViTBlock
from CondConv import CondConv
from codebook import Codebook
import time
import numpy as np
"""cnn+mobileViTblocks+大channels"""
def _make_mit_layer(input_channel,peach_size,transformer_channels,ffn_dim)-> [nn.Sequential, int]:
    block = []

    transformer_dim = transformer_channels
    ffn_dim = ffn_dim
    num_heads = 8
    head_dim = transformer_dim // num_heads

    if transformer_dim % head_dim != 0:
        raise ValueError("Transformer input dimension should be divisible by head dimension. "
                         "Got {} and {}.".format(transformer_dim, head_dim))

    block.append(MobileViTBlock(
        in_channels=input_channel,
        transformer_dim=transformer_dim,
        ffn_dim=ffn_dim,
        n_transformer_blocks=3,
        patch_h=peach_size,
        patch_w=peach_size,
        dropout=0.1,
        ffn_dropout=0.0,
        attn_dropout=0.1,
        head_dim=head_dim,
        conv_ksize=3
    ))

    return nn.Sequential(*block), input_channel

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            #nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            CondConv(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            # CondConv(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class ChannelMerge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1,stride=1),
            # CondConv(in_channels, out_channels, 1, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)



class PDTrans(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super().__init__()
        """img_size=512,in_chans=1,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)"""
        self.img_size=kwargs['img_size']
        self.in_chans = kwargs['in_chans']
        self.sig = nn.Sigmoid()
        # ---------------------------------------------------------------------
        self.left_conv_1 = DoubleConv(self.in_chans, 64)
        self.cm1 = ChannelMerge(67, 64)
        self.down_1 = nn.MaxPool2d(2, 2)
        self.mobileViTBlock1, out_channels = _make_mit_layer(3, 32, 64, 128)  # xxs layer3
        self.mobileViTBlock1_cnn = CondConv(self.in_chans, 12, 3, 1, 1)

        self.left_conv_2 = DoubleConv(64, 128)
        self.cm2 = ChannelMerge(192, 128)
        self.down_2 = nn.MaxPool2d(2, 2)
        self.mobileViTBlock2, out_channels = _make_mit_layer(64, 16, 80, 160)  # xxs layer4
        self.mobileViTBlock2_cnn = CondConv(64, 48, 3, 2, 1)

        self.left_conv_3 = DoubleConv(128, 256)
        self.cm3 = ChannelMerge(384, 256)
        self.down_3 = nn.MaxPool2d(2, 2)
        self.mobileViTBlock3, out_channels = _make_mit_layer(128, 8, 192, 384)  # s layer4
        self.mobileViTBlock3_cnn = CondConv(80, 192, 3, 1, 1)

        self.left_conv_4 = DoubleConv(256, 512)
        self.down_4 = nn.MaxPool2d(2, 2)
        self.mobileViTBlock4, out_channels = _make_mit_layer(256, 4, 240, 480)  # (256, 96, 192)
        self.mobileViTBlock4_cnn = CondConv(80, 768, 3, 1, 1)
        # center
        self.center_conv = DoubleConv(768, 1024)
        self.codebook = Codebook(2048, 2048, beta=0.25)
        self.post_code_conv = torch.nn.Conv2d(1024, 1024, 1)
        # right
        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(768, 512)

        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(384, 256)

        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(192, 128)

        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_5 = nn.Upsample(scale_factor=1.145, mode='bilinear', align_corners=True)
        self.up_6 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.right_conv_4 = DoubleConv(64, 64)

        # output
        self.output = nn.Conv2d(64, 1, 1, 1, 0)
        # self.output = CondConv(64, 1, 1, 1, 0)

    def forward(self, x):
    #stage1:
        #CNN
        y = x.clone()
        x1 = self.left_conv_1(y)
        x1_down = self.down_1(x1)
        x = self.mobileViTBlock1(x)
        t = torch.cat([x1_down, x], dim=1)
        t1 = self.cm1(t)

    # stage2:
        # CNN
        x2 = self.left_conv_2(t1)
        x2_down = self.down_2(x2)
        x2=self.mobileViTBlock2(t1)
        t = torch.cat([x2_down, x2], dim=1)
        t2 = self.cm2(t)

    # stage3:
        # CNN
        x3 = self.left_conv_3(t2)
        x3_down = self.down_3(x3)

    # transformer
        # transformer_block
        x3=self.mobileViTBlock3(t2)
        t = torch.cat([x3_down, x3], dim=1)
        t3 = self.cm3(t)

    # stage4:
        # CNN
        x4 = self.left_conv_4(t3)
        x4_down = self.down_4(x4)

    # transformer
        # transformer_block
        t4 = self.mobileViTBlock4(t3)
        t = torch.cat([x4_down, t4], dim=1)
        x5 = self.center_conv(t)

        codebook_mapping, codebook_indices, q_loss=self.codebook(x5)
        x_code=self.post_code_conv(codebook_mapping)

    # upsempling：
        x6_up = self.up_1(x_code)
        temp = torch.cat((x6_up, t3), dim=1)#t3(256,28,28)
        x6 = self.right_conv_1(temp)

        x7_up = self.up_2(x6)
        temp = torch.cat((x7_up, t2), dim=1)#t2(128,56,56)
        x7 = self.right_conv_2(temp)

        x8_up = self.up_3(x7)
        temp = torch.cat((x8_up, t1), dim=1)#t1(64,112,112)
        x8 = self.right_conv_3(temp)

        x9_up = self.up_4(x8)
        x9 = self.right_conv_4(x9_up)

        # output
        output = self.output(x9)
        outputs= self.sig(output)
        return outputs,q_loss,t1,t2,t3,x5,x6,x7,x8,x9,codebook_mapping,x_code,x6_up


class MATransformerV2(nn.Module):
    def __init__(self):
        super().__init__()
        # left
        self.vit = PDTrans(img_size=512,in_chans=3,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    def forward(self, x):
        # left
        output = self.vit(x)
        return output

