from torch import nn
import torch
from model.dark_net import DarkNet
from model.dark_fpn import DarkFPN
from model.head import Head

class YOLO(nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)
        self.head = Head(num_classes, (width[3], width[4], width[5]))

        img_dummy = torch.zeros(1, 3, 256, 256)
        outputs = self.forward(img_dummy)
        stride_list = []
        for output in outputs:
            stride = 256 / output.shape[-2]
            stride_list.append(stride)
        self.head.stride = torch.tensor(stride_list)
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        x = self.head(list(x))
        return x

if __name__ == "__main__":
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    num_classes = 80

    # 实例化YOLO模型
    model = YOLO(width, depth, num_classes)

    model.eval()

    #print(model)

    # 创建一个假的输入张量来模拟一张256x256的RGB图像
    test_input = torch.randn(1, 3, 640, 480)

    print("### test_input.shape:", test_input.shape)

    # 让模型进行前向传播处理这个输入
    output = model(test_input)
    print("eval output:", output.shape)

    model.train()

    # 让模型进行前向传播处理这个输入
    output = model(test_input)

    print("train output:")

    for out in output:
        print(out.shape)
