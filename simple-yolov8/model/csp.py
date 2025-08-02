from torch import nn
import torch
from model.conv import Conv
from model.bottleneck import Residual

class CSP(nn.Module):
    def __init__(self, in_ch, out_ch, n, add):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, 1, 1)
        self.conv2 = Conv(in_ch, out_ch // 2, 1, 1)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch, 1, 1)
        self.res_m = nn.ModuleList(
                    Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        for m in self.res_m:
            y.append(m(y[-1]))
        y = torch.cat(y, dim=1)
        y = self.conv3(y)
        return y

if __name__ == "__main__":
    # 创建一个随机的输入张量，假设输入有3个通道，图片大小为64x64，批次大小为1
    x = torch.randn(1, 3, 64, 64)

    # 创建CSP模块的实例，输入通道数3，输出通道数16，n设置为3，add设置为True
    model = CSP(3, 16, 3, True)

    # 将输入传递到CSP模块并打印输出结果的形状
    output = model(x)
    print("Output shape:", output.shape)





