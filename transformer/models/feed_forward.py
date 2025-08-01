from torch import nn
import torch

# 实现Transformer中的Feed Forward模块
class FeedForward(nn.Module):
    # 模型的结构定义
    # in_dim是输入特征维度
    # hidden_dim是隐藏层维度
    def __init__(self, in_dim, hidden_dim, drop_prob):
        super().__init__()
        # 在init函数中，定义了Feed Forward模型中的结构
        # 前馈神经网络
        # 第1个线性层的输入特征数为in_dim
        # 第2个线性层的输出特征数也为in_dim
        # 因此ffn不会修改输入数据的特征维度
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 第1个线性层
            nn.ReLU(),  # relu激活函数
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, in_dim),  # 第2个线性层
        )

    # 前向传播计算
    def forward(self, x):
        x = self.ffn(x)  # 将x输入至前馈神经网络ffn中进行计算
        return x  # 返回


# 在main函数中编写测试代码
if __name__ == '__main__':
    in_dim = 512  # 输入维度，注意力机制层的输出维度
    hidden_dim = 2048  # FeedForward的隐藏层维度
    drop_prob = 0.1  # 丢弃比率为0.1

    # 定义FeedForward模型
    model = FeedForward(in_dim, hidden_dim, drop_prob)
    print(model)  # 将其打印

    # 输入数据x
    # 表示了32个数据
    # 每个数据最多有10个单词
    # 每个单词的维度是in_dim
    # 张量x是FeedForward层的输入，也就是注意力机制层的输出
    x = torch.rand(32, 10, in_dim)
    # 打印x的尺寸
    print(f"x.shape = {x.shape}")

    # 将x输入至model，计算x经过feedforward的结果
    output = model(x)
    # 打印output的尺寸
    print(f"output.shape = {output.shape}")

