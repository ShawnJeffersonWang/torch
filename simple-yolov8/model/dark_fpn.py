from torch import nn
import torch
from model.conv import Conv
from model.csp import CSP

class DarkFPN(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6

if __name__ == "__main__":
    # 假定 width 和 depth 参数根据模型设计已经定义
    width = [3, 32, 64, 128, 256, 512]
    depth = [1, 2, 2]
    model = DarkFPN(width, depth)
    # 创建随机输入数据
    # 创建随机输入数据
    p3 = torch.randn(1, width[3], 64, 64)  # 假设 p3 特征图大小为 64x64
    p4 = torch.randn(1, width[4], 32, 32)  # 假设 p4 特征图大小为 32x32
    p5 = torch.randn(1, width[5], 16, 16)  # 假设 p5 特征图大小为 16x16
    outputs = model((p3, p4, p5))
    # 打印输出的特征图尺寸
    print("输出的特征图尺寸:")
    for output in outputs:
        print(output.shape)


