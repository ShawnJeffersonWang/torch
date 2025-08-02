from torch import nn

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_ch,
                              out_channels = out_ch,
                              kernel_size = k_size,
                              stride = stride,
                              padding = k_size // 2,
                              bias = False)
        self.norm = nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

import torch

if __name__ == "__main__":
    # 创建一个随机的输入张量，假设输入有3个通道，图片大小为64x64，批次大小为1
    x = torch.randn(4, 3, 3, 3)

    # 创建一个卷积层实例，输入通道3，输出通道16，卷积核大小为3，步长为1
    conv_layer = Conv(3, 2, 2, 1)

    # 将输入传递到卷积层并打印输出结果的形状
    output = conv_layer(x)
    print("Output shape:", output.shape)















