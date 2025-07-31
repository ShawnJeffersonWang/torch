import torch
from torch import nn

# 定义优化后的神经网络
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 线性层1，输入层到隐藏层之间的线性层
        self.layer1 = nn.Linear(784, 256)
        # Dropout层，减少过拟合
        self.dropout1 = nn.Dropout(p=0.5)
        # 线性层2，隐藏层到隐藏层之间的线性层
        self.layer2 = nn.Linear(256, 128)
        # Dropout层，减少过拟合
        self.dropout2 = nn.Dropout(p=0.3)
        # 线性层3，隐藏层到输出层的线性层
        self.layer3 = nn.Linear(128, 10)

    # 在前向传播，forward函数中，输入为图像x
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 使用view函数，将x展平
        x = self.layer1(x)  # 输入至layer1
        x = torch.relu(x)  # 使用relu激活
        x = self.dropout1(x)  # Dropout层1
        x = self.layer2(x)  # 输入至layer2
        x = torch.relu(x)  # 使用relu激活
        x = self.dropout2(x)  # Dropout层2
        return self.layer3(x)  # 输入至layer3，计算输出结果


# 打印模型的参数
def print_parameters(model):
    cnt = 0
    for name, layer in model.named_children():  # 遍历每一层
        print(f"layer({name}) parameters:")
        for p in layer.parameters():
            print(f'\t {p.shape} has {p.numel()} parameters')
            cnt += p.numel()  # 累加参数数量
    print('The model has %d trainable parameters\n' % (cnt))


# 打印输入张量x经过每一层时的维度变化情况
def print_forward(model, x):
    print(f"x: {x.shape}")  # 输入形状
    x = x.view(-1, 28 * 28)  # 展平
    print(f"after view: {x.shape}")
    x = model.layer1(x)  # 经过第1个线性层
    print(f"after layer1: {x.shape}")
    x = torch.relu(x)  # relu激活
    print(f"after relu(layer1): {x.shape}")
    x = model.dropout1(x)  # Dropout1
    print(f"after dropout1: {x.shape}")
    x = model.layer2(x)  # 经过第2个线性层
    print(f"after layer2: {x.shape}")
    x = torch.relu(x)  # relu激活
    print(f"after relu(layer2): {x.shape}")
    x = model.dropout2(x)  # Dropout2
    print(f"after dropout2: {x.shape}")
    x = model.layer3(x)  # 经过第3个线性层
    print(f"after layer3: {x.shape}")


if __name__ == '__main__':
    model = Network()  # 定义优化后的模型
    print(model)  # 打印模型结构
    print("")

    print_parameters(model)  # 打印模型参数
    # 打印输入张量x经过每一层的维度变化情况
    x = torch.zeros([5, 28, 28])
    print_forward(model, x)
