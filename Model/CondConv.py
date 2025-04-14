from torch import nn as nn
import torch
from torch.nn import functional as F
# 输入为 [N, C, H, W]，需要两个参数，in_planes为输特征通道数，K 为专家个数
class Attention(nn.Module):
    def __init__(self,in_planes,K):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.net=nn.Conv2d(in_planes, K, kernel_size=1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        # 将输入特征全局池化为 [N, C, 1, 1]
        att=self.avgpool(x)
        # 使用1X1卷积，转化为 [N, K, 1, 1]
        att=self.net(att)
        # 将特征转化为二维 [N, K]
        att=att.view(x.shape[0],-1)
        # 使用 sigmoid 函数输出归一化到 [0,1] 区间
        return self.sigmoid(att)


class CondConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding,
                 groups=1,K=4):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.K = K
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Attention(in_planes=in_planes,K=K)
        self.weight = nn.Parameter(torch.randn(K,out_planes,in_planes//groups,
                                             kernel_size,kernel_size),requires_grad=True)

    def forward(self,x):
        # 调用 attention 函数得到归一化的权重 [N, K]
        N,in_planels, H, W = x.shape
        softmax_att=self.attention(x)
        # 把输入特征由 [N, C_in, H, W] 转化为 [1, N*C_in, H, W]
        x=x.view(1, -1, H, W)

        # 生成随机 weight [K, C_out, C_in/groups, 3, 3]
        # 注意添加了 requires_grad=True，这样里面的参数是可以优化的
        weight = self.weight
        # 改变 weight 形状为 [K, C_out*(C_in/groups)*3*3]
        weight = weight.view(self.K, -1)

        # 矩阵相乘：[N, K] X [K, C_out*(C_in/groups)*3*3] = [N, C_out*(C_in/groups)*3*3]
        aggregate_weight = torch.mm(softmax_att,weight)
        # 改变形状为：[N*C_out, C_in/groups, 3, 3]，即新的卷积核权重
        aggregate_weight = aggregate_weight.view(
            N*self.out_planes, self.in_planes//self.groups,
            self.kernel_size, self.kernel_size)
        # 用新生成的卷积核进行卷积，输出为 [1, N*C_out, H, W]
        output=F.conv2d(x,weight=aggregate_weight,
                        stride=self.stride, padding=self.padding,
                        groups=self.groups*N)
        # 形状恢复为 [N, C_out, H, W]
        output=output.view(N, self.out_planes, int(H/self.stride), int(W/self.stride))
        return output