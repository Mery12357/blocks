# --------------------------------------------------------
# 论文：DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention
# GitHub地址：https://github.com/cecret3350/DEA-Net/tree/main
# --------------------------------------------------------
'''
DEA-Net：基于细节增强卷积和内容引导注意力的单图像去雾 (IEEE TIP 2024顶会论文)

我们提出了一种新的注意机制，称为轮廓引导注意（CGA），以一种从粗到细的方式生成特定于通道的sim。
通过使用输入的特征引导SIM的生成，CGA为每个通道分配唯一的SIM，
使模型关注每个通道的重要区域。因此，可以强调用特征编码的更多有用的信息，以有效地提高性能。
此外，还提出了一种基于cga的混合融合方案，可以有效地将编码器部分的低级特征与相应的高级特征进行融合。
'''
import torch
from torch import nn
from einops.layers.torch import Rearrange

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

# 双分支特征融合
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = CGAFusion(32)
    input1 = torch.rand(3, 32, 64, 64)
    input2 = torch.rand(3, 32, 64, 64)
    output = block(input1, input2)
    print(output.size())
