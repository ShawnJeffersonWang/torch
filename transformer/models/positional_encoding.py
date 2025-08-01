import torch
import torch.nn as nn
import math

# 位置编码模块
class PositionalEncoding(nn.Module):
    # 传入词向量的维度embed_dim和最大位置长度max_pos
    def __init__(self, embed_dim, max_pos):
        super(PositionalEncoding, self).__init__()
        # 定义保存位置编码的数组
        PE = torch.zeros(max_pos, embed_dim)
        # 生成从0到max_pos-1的位置数组pos
        pos = torch.arange(0, max_pos).unsqueeze(1).float()

        # 从0开始，生成间隔为2的序列，对应公式中的2i
        multi_term = torch.arange(0, embed_dim, 2).float()
        # 计算公式中pos对应的系数部分
        # 这里计算的是e^(2i * (-log(10000/d)))
        # 它从数学计算上等价于1 / (10000^(2i/d))
        multi_term = torch.exp(multi_term * (-math.log(10000.0) / embed_dim))
        # 使用正弦函数sin和余弦函数cos，生成位置编码数组PE
        PE[:, 0::2] = torch.sin(pos * multi_term)
        PE[:, 1::2] = torch.cos(pos * multi_term)

        # 将数组PE注册为一个不需要梯度更新的缓存数组
        # 相当于将位置信息保存在了一个常量数组中
        self.register_buffer('PE', PE.unsqueeze(0))

    # 前向传播函数，函数传入输入数据x
    def forward(self, x):
        # 将x加上位置信息PE，得到添加位置信息的词向量
        return x + self.PE[:, :x.size(1)].clone().detach()

if __name__ == "__main__":
    embed_dim = 4  # 词向量维度
    max_pos = 10  # 最大序列长度

    # 定义位置编码模型
    model = PositionalEncoding(embed_dim, max_pos)

    # 输入数据，代表了2个样本，每个样本长度是5，每个词的维度是embed_dim
    x = torch.zeros(2, 5, embed_dim)

    # 将x传入模型model，计算添加位置信息的结果output
    output = model(x)

    print("x:")
    print(x.shape) # 打印x的尺寸
    print("PE:")
    print(model.PE.shape) # 打印PE数组的尺寸
    print("output:")
    print(output.shape) # 打印output的尺寸

