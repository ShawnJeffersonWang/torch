from torch import nn
from torchvision import models

class Vgg16Loss(nn.Module): # 损失计算网络
    # 预训练的vgg16模型路径model_path
    def __init__(self, model_path):
        super(Vgg16Loss, self).__init__()
        model = models.vgg16().eval() # 定义原始的VGG16模型
        # 加载预训练的VGG16模型
        model.load_state_dict(torch.load(model_path))
        # 定义seq1到seq4，四个Sequential序列
        self.seq1 = nn.Sequential()
        self.seq2 = nn.Sequential()
        self.seq3 = nn.Sequential()
        self.seq4 = nn.Sequential()
        for x in range(4): # 保存0-3层
            self.seq1.add_module(str(x), model.features[x])
        for x in range(4, 9): # 保存4-8层
            self.seq2.add_module(str(x), model.features[x])
        for x in range(9, 16): # 保存9-15层
            self.seq3.add_module(str(x), model.features[x])
        for x in range(16, 23): # 保存16-22层
            self.seq4.add_module(str(x), model.features[x])

        for param in self.parameters():
            # 将当前模型中的全部参数固定住
            param.requires_grad = False

    def forward(self, x):
        # 按照seq1到seq4的顺序，计算输入数据x
        x = self.seq1(x)
        res1 = x
        x = self.seq2(x)
        res2 = x
        x = self.seq3(x)
        res3 = x
        x = self.seq4(x)
        res4 = x
        # 定义一个字典，字典的key是relu对应的层名，value是res1到res4
        out = {
            'relu1_2': res1,
            'relu2_2': res2,
            'relu3_3': res3,
            'relu4_3': res4
        }
        return out

import torch

if __name__ == '__main__':
    images = torch.randn(5, 3, 224, 224) # 随机数据
    vgg_loss = Vgg16Loss('./pretrained/hub/checkpoints/vgg16-397923af.pth')
    out = vgg_loss(images) # 计算输入数据
    # 打印输出结果
    for key, value in out.items():
        print(f"{key}: {value.shape}")

