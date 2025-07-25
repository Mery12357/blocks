from math import sqrt
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

# 论文：https://arxiv.org/pdf/2401.00423
# 代码：https://github.com/YoZhibo/MSGNet/tree/main

'''
题目：MSGNet：学习多变量时间序列预测的多尺度序列间相关性   AAAI 2024 顶会
时间序列预测任务即插即用模块：ScaleGraphBlock

在理解多个时间序列之间不同时间尺度上不同的序列间相关性方面仍然存在重大的研究差距，
这一领域在文献中受到的关注有限。为了弥补这一差距，本文引入了 MSGNet，
这是一种先进的深度学习模型，旨在使用频域分析和自适应图卷积来捕获多个时间尺度上不同的序列间相关性。
在几个数据集上进行了广泛的实验，以展示MSGNet的有效性.

MSGNet 采用了多个 ScaleGraphBlock 模块，每个模块包含三个关键模块：
      (1)用于多尺度数据识别的 FFT 模块、
      (2)用于在时间尺度内进行序列间相关性学习的自适应图形卷积模块，
      (3)用于序列内相关性学习的多头注意力模块。

'''
class Configs:
    def __init__(self):
        self.seq_len = 100      # 输入序列长度
        self.pred_len = 24      # 预测长度
        self.top_k = 3          # k值
        self.d_model = 64       # 模型维度
        self.d_ff = 256         # FFN中的隐藏层大小
        self.n_heads = 4        # 多头注意力机制中的头数
        self.dropout = 0.1      # dropout率
        self.c_out = 64         # 图卷积输出通道数
        self.conv_channel = 64  # 卷积通道数
        self.skip_channel = 32  # 跳过连接通道数
        self.gcn_depth = 2      # GCN的深度
        self.propalpha = 0.05   # 图卷积中的propalpha系数
        self.node_dim = 10      # 节点维度
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        # x = torch.einsum('ncwl,wv->nclv',(x,A)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)
class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho
class self_attention(nn.Module):
    def __init__(self, attention, d_model ,n_heads):
        super(self_attention, self).__init__()
        d_keys =  d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention( attention_dropout = 0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries ,keys ,values, attn_mask= None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
                    queries,
                    keys,
                    values,
                    attn_mask
                )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out , attn
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # return V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
class Attention_Block(nn.Module):
    def __init__(self,  d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)

class GraphBlock(nn.Module):
    def __init__(self, c_out , d_model , conv_channel, skip_channel,
                        gcn_depth , dropout, propalpha ,seq_len , node_dim):
        super(GraphBlock, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        self.start_conv = nn.Conv2d(1 , conv_channel, (d_model - c_out + 1, 1))
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, seq_len , (1, seq_len ))
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)

    # x in (B, T, d_model)
    # Here we use a mlp to fit a complex mapping f (x)
    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x.unsqueeze(1).transpose(2, 3)
        out = self.start_conv(out)
        out = self.gelu(self.gconv1(out , adp))
        out = self.end_conv(out).squeeze()
        out = self.linear(out)

        return self.norm(x + out)

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]
class ScaleGraphBlock(nn.Module):
    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.att0 = Attention_Block(configs.d_model, configs.d_ff,
                                   n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(
                GraphBlock(configs.c_out , configs.d_model , configs.conv_channel, configs.skip_channel,
                        configs.gcn_depth , configs.dropout, configs.propalpha ,configs.seq_len,
                           configs.node_dim))
    def forward(self, x):
        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            scale = scale_list[i]
            #Gconv
            x = self.gconv[i](x)
            # paddng
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // scale, scale, N)
        #for Mul-attetion
            out = out.reshape(-1 , scale , N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1 , scale , N).reshape(B ,-1 ,N)
        # #for simpleVIT
        #     out = self.att(out.permute(0, 3, 1, 2).contiguous()) #return
        #     out = out.permute(0, 2, 3, 1).reshape(B, -1 ,N)

            out = out[:, :self.seq_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        # residual connection
        res = res + x
        return res

if __name__ == '__main__':
    configs = Configs()
    # 创建一个 ScaleGraphBlock实例
    model = ScaleGraphBlock(configs)
    # 随机创建输入模拟数据   B L N
    batch_size = 8
    d_model = 64
    input = torch.randn(batch_size, configs.seq_len, d_model)
    # 执行前向传播
    output = model(input)
    # 输出结果形状
    print("输入张量形状:", input.shape)
    print("输出张量形状:", output.shape)




