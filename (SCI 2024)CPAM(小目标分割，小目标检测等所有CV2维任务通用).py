import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
即插即用模块：CPAM通道和位置注意力模块  SCI 2024

CPAM（通道和位置注意力机制）模块的主要作用是提升小目标（如细胞）分割任务中的精度。

通道注意力：CPAM首先执行通道注意力机制。它对每个通道进行全局平均池化，
然后通过1D卷积来捕捉通道之间的交互信息。这种方法避免了降维问题，
确保模型能够有效地聚焦在最相关的通道特征上。
位置注意力：在完成通道注意力后，CPAM对特征图的水平和垂直轴进行位置注意力处理，
通过池化操作获取空间结构信息。这一步有助于更准确地定位小目标的空间位置。

通过同时优化通道和位置信息，CPAM能够帮助模型更好地识别和分割小而密集的目标，提高分割精度和检测效果.
适用于：小目标分割，小目标检测等所有CV2维任务通用多尺度特征提取模块

'''
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv_l_post_down = Conv(in_dim, 2*in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms
class SSFF(nn.Module):
    def __init__(self, inc, channel):
        super(SSFF, self).__init__()
        self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
class CPAM(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x
# CPAM通道位置注意力模块：专注于信息通道和空间位置相关的小物体，以提高检测和分割性能
if __name__ == '__main__':
    input1 = torch.randn(1, 256, 32, 32)
    input2 = torch.randn(1, 256, 32, 32)
    model = CPAM(ch=256)
    output = model([input1, input2])
    print('input1_size:', input1.size())
    print('input2_size:', input2.size())
    print('output_size:', output.size())

