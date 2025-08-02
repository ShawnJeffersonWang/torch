import torch
from torch import nn
from model.conv import Conv

class SPP(nn.Module):
    def __init__(self, in_ch, out_ch, k_size):
        #in_ch: 输入通道数,out_ch: 输出通道数,k_size: 池化核大小
        super().__init__()
        # 第一个1x1卷积，用于降维
        self.conv1 = Conv(in_ch, in_ch // 2, 1, 1)
        # 最后的1x1卷积，用于调整通道数
        self.conv2 = Conv(in_ch * 2, out_ch, 1, 1)
        # 最大池化层，padding设置为k_size//2保持特征图尺寸不变
        self.res_m = nn.MaxPool2d(k_size, 1, k_size // 2)

    def forward(self, x):
        # x: 输入特征图, out: 经过SPP处理后的特征图
        x = self.conv1(x) # 先通过1x1卷积降维
        y1 = self.res_m(x) # 第一级池化
        y2 = self.res_m(y1) # 第二级池化
        # 将原始特征和不同尺度的池化结果拼接
        out = torch.cat([x, y1, y2, self.res_m(y2)], 1)
        out = self.conv2(out) # 通过1x1卷积调整通道数
        return out


if __name__ == "__main__":
    # 测试代码
    # 创建输入张量 (batch_size=1, channels=64, height=20, width=20)
    x = torch.randn(1, 64, 20, 20)
    # 实例化SPP模块 (输入64通道，输出32通道，池化核大小5)
    spp = SPP(in_ch=64, out_ch=32, k_size=5)
    # 前向传播
    out = spp(x)
    # 打印输入输出尺寸
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {out.shape}")
