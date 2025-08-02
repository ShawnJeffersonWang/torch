from torch import nn
import torch
class DFL(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        x = x.softmax(1)
        x = self.conv(x)
        x = x.view(b, 4, a)
        return x

if __name__ == "__main__":
    ch = 4  # 通道数设置为 4，对应模型的设计
    model = DFL(ch)
    # 创建随机输入数据，这里假设批量大小为1，总通道数为16，长度为10
    x = torch.randn(1, 16, 10)
    output = model(x)
    # 打印输出结果
    print("输出的数据形状:", output.shape)
