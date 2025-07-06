import torch
import torch.nn as nn
"""
    论文地址：https://arxiv.org/pdf/2406.03751
    论文题目：Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting（AAAI 2025）
    中文题目：用于时间序列预测的自适应多尺度分解框架(AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1aSckeYEpj/
    多尺度分解混合块（Multi-Scale Decomposable Mixing Block, MDM）
         发现问题：时间序列呈现出粗-细粒度时间模式，粗粒度模式反映宏观背景，细粒度模式提供微观反馈，这些互补信息尺度共同为时间序列提供全面视角。
         解决思路：将时间序列分解为单独的时间模式，然后将它们混合起来，以增强时间序列数据。
"""
class MDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        super(MDM, self).__init__()
        self.seq_len = input_shape[0]  # 序列长度
        self.k = k  # 池化层数
        if self.k > 0:
            # 计算每层的池化核大小
            self.k_list = [c ** i for i in range(k, 0, -1)]
            # 创建平均池化层列表
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])
            # 创建线性层列表
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.seq_len // k, self.seq_len // k),
                                  nn.GELU(),
                                  nn.Linear(self.seq_len // k, self.seq_len * c // k),
                                  )
                    for k in self.k_list
                ]
            )
        self.layernorm = layernorm  # 是否使用层归一化
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])  # 批归一化层

    def forward(self, x):
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)  # 归一化输入
        if self.k == 0:
            return x  # 如果k为0，直接返回输入
        # 输入形状：[batch_size, feature_num, seq_len]
        sample_x = []

        for i, k in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](x))  # 对输入进行平均池化 【将时间序列分解为单独的时间模式】

        sample_x.append(x)  # 添加原始输入
        n = len(sample_x)
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])  # 通过线性层
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)  # 加上残差 【然后将它们混合起来，以增强时间序列数据。】

        # 返回最终输出：[batch_size, feature_num, seq_len]
        return sample_x[n - 1]

if __name__ == "__main__":
    input_shape = (48, 10)  # 示例：序列长度=48，特征数量=10
    # 初始化MDM模型
    model = MDM(input_shape=input_shape)
    # 创建一些示例输入数据
    batch_size = 16
    x = torch.randn(batch_size, input_shape[1], input_shape[0])  # 输入形状：[batch_size, feature_num, seq_len]
    # 通过模型进行前向传递
    output = model(x)
    # 打印输出形状
    print("Output shape:", x.shape)
    print("Output shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")