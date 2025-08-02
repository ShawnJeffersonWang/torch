from torch import nn
import torch
from model.conv import Conv  # 假设Conv是你自定义的卷积模块

class Residual(nn.Module):
    def __init__(self, ch, add):
        """
        残差模块
        参数：
        ch: 输入和输出的通道数
        add: 是否添加残差连接（True表示添加，即输出为 F(x) + x）
        """
        super().__init__()
        self.add_m = add
        self.res_m = nn.Sequential(
            Conv(ch, ch, 3, 1),  # 第一个卷积层
            Conv(ch, ch, 3, 1)  # 第二个卷积层
        )

    def forward(self, x):
        """
        前向传播
        如果 add_m 为 True，则返回 F(x) + x；否则只返回 F(x)
        """
        if self.add_m:
            return self.res_m(x) + x
        return self.res_m(x)


# 测试代码
if __name__ == "__main__":
    # 创建一个测试输入张量（批大小为1，通道数为16，尺寸为64x64）
    x = torch.randn(1, 16, 64, 64)

    # 创建一个Residual模块，通道数为16，使用残差连接
    res_block = Residual(16, add=True)

    # 前向传播并打印输出形状
    out = res_block(x)
    print("输出形状:", out.shape)  # 应该仍为 (1, 16, 64, 64)

