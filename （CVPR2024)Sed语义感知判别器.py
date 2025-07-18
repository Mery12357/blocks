import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from inspect import isfunction
import math
from torch import nn, einsum
from einops import rearrange, repeat
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F
import functools
import math
import clip
from clip.model import ModifiedResNet
import ModifiedSpatialTransformer


# https://arxiv.org/abs/2402.19387
# https://github.com/lbc12345/SeD/tree/main
'''
# SeD: Semantic-Aware Discriminator for Image Super-Resolution  CVPR2024
# SeD：用于图像超分辨率的语义感知判别器

论文的主要贡献：
本文介绍了一种改进的图像超分辨率（SR）生成对抗网络（GAN），称为语义感知判别器（SeD）。
传统的GAN在学习图像生成时容易生成虚假纹理，而SeD通过引入图像的语义信息来缓解这个问题。

实验表明，这种方法在图像超分辨率（SR）和Real SR两个任务中表现出色。
SeD (Semantic-aware Discriminator) 被设计成一个块判别器（PatchGAN）的原因有几个方面，
主要是为了更好地捕捉图像中的局部细节和语义信息。

为什么构造 Construct a PatchGAN discriminator 结构？
下面详细解释一下为什么选择这种（PatchGAN）结构以及它的优势。
捕获局部纹理：PatchGAN判别器 的设计允许它专注于图像的小区域，这样可以更好地捕捉局部细节和纹理，
           这对于超分辨率任务非常重要，因为恢复高清图像中的细腻纹理是关键。
计算效率成本低：PatchGAN判别器 相较于全图判别器（即图像级别的判别器），计算成本更低。
        它不需要处理整个图像，而是关注图像的局部区域，这有助于加速训练过程。
在 GAN 训练中，有一个常见问题，即生成器可能只学会生成有限种类的样本。

PatchGAN 有助于减轻这一问题，因为它关注的是图像的局部区域而不是整体，从而能够更好地区分真假样本。

SeD集成到大多数基于GAN生成网络的图像超分辨率生成方法中，并取得了出色的性能。
'''
class CLIP_Semantic_extractor(ModifiedResNet):
    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32):
        super(CLIP_Semantic_extractor, self).__init__(layers=layers, output_dim=output_dim, heads=heads)

        ckpt = 'RN50' if path is None else path

        if pretrained:
            model, _ = clip.load(ckpt, device='cpu')

        self.load_state_dict(model.visual.state_dict())
        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )
        self.requires_grad_(False)

        del model

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = (x - self.mean) / self.std
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class SeD_P(nn.Module):
    def __init__(self, input_nc, ndf=64, semantic_dim=1024, semantic_size=16, use_bias=True, nheads=1, dhead=64):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()

        kw = 4
        padw = 1
        # ss = [128, 64, 32, 31, 30]  # PatchGAN's spatial size
        # cs = [64, 128, 256, 512, 1]  # PatchGAN's channel size

        norm = spectral_norm

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.conv_first = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)

        self.conv1 = norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = math.ceil(64 / semantic_size)
        self.att1 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=128,
                                               up_factor=upscale)

        ex_ndf = int(semantic_dim / upscale ** 2)
        self.conv11 = norm(nn.Conv2d(ndf * 2 + ex_ndf, ndf * 2, kernel_size=3, stride=1, padding=padw, bias=use_bias))

        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = math.ceil(32 / semantic_size)
        self.att2 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=256,
                                               up_factor=upscale)

        ex_ndf = int(semantic_dim / upscale ** 2)
        self.conv21 = norm(nn.Conv2d(ndf * 4 + ex_ndf, ndf * 4, kernel_size=3, stride=1, padding=padw, bias=use_bias))

        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        upscale = math.ceil(31 / semantic_size)
        self.att3 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=512,
                                               up_factor=upscale, is_last=True)

        ex_ndf = int(semantic_dim / upscale ** 2)
        self.conv31 = norm(nn.Conv2d(ndf * 8 + ex_ndf, ndf * 8, kernel_size=3, stride=1, padding=padw, bias=use_bias))

        self.conv_last = nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)

        init_weights(self, init_type='normal')

    def forward(self, input, semantic):
        """Standard forward."""
        input = self.conv_first(input)
        input = self.lrelu(input)

        input = self.conv1(input)
        se = self.att1(semantic, input)
        input = self.lrelu(self.conv11(torch.cat([input, se], dim=1)))

        input = self.conv2(input)
        se = self.att2(semantic, input)
        input = self.lrelu(self.conv21(torch.cat([input, se], dim=1)))

        input = self.conv3(input)
        se = self.att3(semantic, input)
        input = self.lrelu(self.conv31(torch.cat([input, se], dim=1)))

        input = self.conv_last(input)
        return input

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def exists(val):
    return val is not None

def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        # query: semantic; key, value: image.

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # x = self.attn1(self.norm1(x)) + x
        # x = self.attn2(self.norm2(x), context=context) + x
        # x = self.ff(self.norm3(x)) + x
        x = self.attn1(self.norm1(x))
        x = self.attn2(self.norm2(x), context=context)
        x = self.ff(self.norm3(x))
        return x

class ModifiedSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=1024, up_factor=2, is_last=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_context = nn.Conv2d(context_dim,
                                      inner_dim,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=inner_dim)
             for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        up_channels = int(in_channels / up_factor / up_factor)
        if not is_last:
            self.conv_out = nn.Conv2d(up_channels, up_channels, 3, 1, 1)
        else:
            self.conv_out = nn.Conv2d(up_channels, up_channels, 4, 1, 1)
        self.up_factor = up_factor

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        # x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        context = self.proj_context(context)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        # return x + x_in
        x = rearrange(x, 'b (c uw uh) h w -> b c (h uh) (w uw)', uh=self.up_factor,
                      uw=self.up_factor).contiguous()  # rescale the semantic
        x = self.conv_out(x)
        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # s = CLIP_Semantic_extractor()
    # input_data = torch.randn(1, 3, 256, 256)  # 要求输入图像大小为 256*256
    # print(s(input_data).size())

    input_data = torch.randn(1, 3, 256, 256)  # 要求输入图像大小为 256*256
    Semantic_extractor = CLIP_Semantic_extractor()
    semantic_data = Semantic_extractor(input_data) # 语义信息大小为 (1,1024, 16, 16)

    # 创建 SeD_P模型实例
    """Construct a PatchGAN discriminator 构建一个 PatchGAN 判别器 
           Parameters:
               input_nc (int)  -- the number of channels in input images
               ndf (int)       -- the number of filters in the last conv layer
               n_layers (int)  -- the number of conv layers in the discriminator
               norm_layer      -- normalization layer
    """
    model = SeD_P(input_nc=3, semantic_dim=1024, semantic_size=16)
    # 执行前向传播
    output = model(input_data, semantic=semantic_data)

    print('PatchGAN_size:',output.size())