from torch import nn
import torch
from model.conv import Conv  # 自定义卷积模块
from model.csp import CSP  # 跨阶段部分连接模块
from model.spp import SPP  # 空间金字塔池化模块

class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        # 第一阶段 (p1): 初始下采样
        self.p1 = nn.Sequential(
            Conv(width[0], width[1], 3, 2)  # 3x3卷积，stride=2下采样
        )
        # 第二阶段 (p2): 下采样+CSP模块
        self.p2 = nn.Sequential(
            Conv(width[1], width[2], 3, 2),  # 3x3卷积，stride=2下采样
            CSP(width[2], width[2], depth[0], True)  # CSP模块
        )
        # 第三阶段 (p3): 下采样+CSP模块 (此阶段输出用于检测)
        self.p3 = nn.Sequential(
            Conv(width[2], width[3], 3, 2),  # 3x3卷积，stride=2下采样
            CSP(width[3], width[3], depth[1], True)  # CSP模块
        )
        # 第四阶段 (p4): 下采样+CSP模块 (此阶段输出用于检测)
        self.p4 = nn.Sequential(
            Conv(width[3], width[4], 3, 2),  # 3x3卷积，stride=2下采样
            CSP(width[4], width[4], depth[2], True)  # CSP模块
        )
        # 第五阶段 (p5): 下采样+CSP模块+SPP (此阶段输出用于检测)
        self.p5 = nn.Sequential(
            Conv(width[4], width[5], 3, 2),  # 3x3卷积，stride=2下采样
            CSP(width[5], width[5], depth[0], True),  # CSP模块
            SPP(width[5], width[5], 5)  # 空间金字塔池化
        )
    def forward(self, x):
        p1 = self.p1(x)  # 第一阶段
        p2 = self.p2(p1)  # 第二阶段
        p3 = self.p3(p2)  # 第三阶段 (输出1)
        p4 = self.p4(p3)  # 第四阶段 (输出2)
        p5 = self.p5(p4)  # 第五阶段 (输出3)
        return p3, p4, p5  # 返回三个不同尺度的特征图


if __name__ == "__main__":
    # 测试代码

    # 创建测试输入 (batch_size=1, channels=3, height=640, width=480)
    test_input = torch.randn(1, 3, 640, 480)

    # 初始化DarkNet模型
    # width: [输入通道, p1通道, p2通道, p3通道, p4通道, p5通道]
    # depth: [p2的CSP重复次数, p3的CSP重复次数, p4的CSP重复次数]
    model = DarkNet(width=[3, 16, 32, 64, 128, 256], depth=[1, 2, 2])

    # 前向传播
    outputs = model(test_input)

    # 打印各阶段输出形状
    print("输入形状:", test_input.shape)
    for i, output in enumerate(outputs, 3):  # 从阶段3开始编号
        print(f"阶段 {i} 输出形状: {output.shape}")

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

