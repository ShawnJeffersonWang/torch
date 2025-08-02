import torch
from torch import nn

if __name__ == "__main__":
    # 创建一个随机的输入张量，假设输入有3个通道，图片大小为3x3，批次大小为1
    x = torch.randn(4, 3, 3, 3)

    conv1 = nn.Conv2d(in_channels=3,
              out_channels=2,
              kernel_size=2,
              stride=1,
              padding=0,
              bias=False)

    # 将输入传递到卷积层并打印输出结果的形状
    output = conv1(x)
    print("Output shape:", output.shape)

    conv2 = nn.Conv2d(in_channels=3,
                      out_channels=2,
                      kernel_size=2,
                      stride=1,
                      padding=1,
                      bias=False)

    # 将输入传递到卷积层并打印输出结果的形状
    output = conv2(x)
    print("Output shape:", output.shape)

