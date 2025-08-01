import math
import torch
from torch import nn
import torch.nn.functional as F

# 实现一个split_tensor的函数
# 用于将线性层计算后得到的张量tensor，拆分为多头的形式
def split_tensor(tensor, head_num):
    # 输入的张量尺寸: [batch_size×seq_length×d_model]
    # 将张量tensor，变换为: batch_size×head_num×seq_length×head_dim
    batch_size, seq_length, d_model = tensor.size()
    assert d_model % head_num == 0
    head_dim = d_model // head_num # 每个头的维度
    # 使用view函数，对张量维度进行变换
    tensor = tensor.view(batch_size, seq_length, head_num, head_dim)
    # 使用transpose交换张量维度
    tensor = tensor.transpose(1, 2)
    return tensor # 返回变换后的张量

# 根据公式计算缩放点积注意力，函数传入q、k、v三组张量和掩码标记mask
def scale_dot_product_attention(q, k, v, mask=None):
    # 通过张量k，获取参数
    _, _, _, head_dim = k.size()
    # 对张量k转置，转置最后两个维度
    k = k.transpose(2, 3)
    # 计算q和k转置的相关性数组
    score = (q @ k) / math.sqrt(head_dim)

    if mask is not None: # 检查掩码数组是否为空
        # 将无效的数据进行掩码
        score = score.masked_fill(mask == 0, -1e9)

    # 相关性数组的概率形式
    score = F.softmax(score, dim=-1)
    return score @ v # 将score和v两个矩阵相乘

# 用于将多头注意力的多个张量结果合并为一个整体张量
def concat_tensor(tensor):
    # 获取输入张量tensor的尺寸信息，包括4个维度
    batch_size, head_num, seq_length, head_dim = tensor.size()
    tensor = tensor.transpose(1, 2)
    tensor = tensor.reshape(batch_size, seq_length, head_num * head_dim)
    return tensor # 返回合并后的张量

class MultiHeadAttention(nn.Module):
    # 模型的初始化函数，传入模型的内部维度和多头的数量
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        # 分别定义4组线性层
        self.q_net = nn.Linear(d_model, d_model)
        self.k_net = nn.Linear(d_model, d_model)
        self.v_net = nn.Linear(d_model, d_model)
        self.concat_net = nn.Linear(d_model, d_model)

    # 模型的推理函数
    def forward(self, q, k, v, mask=None):
        #1.使用线性层对3个输入张量q、k、v，进行特征变换
        q = self.q_net(q)
        k = self.k_net(k)
        v = self.v_net(v)
        # 2.基于多头的数量，拆分q、k、v三个张量
        q = split_tensor(q, self.n_head)
        k = split_tensor(k, self.n_head)
        v = split_tensor(v, self.n_head)
        # 3.基于attention公式，计算缩放点积运算
        out = scale_dot_product_attention(q, k, v, mask=mask)
        # 4.将多头结果进行拼接，并使用concat_net计算最终结果
        out = concat_tensor(out)
        out = self.concat_net(out)
        return out

# 测试代码
if __name__ == "__main__":
    # 模拟输入张量
    batch_size = 2
    seq_length = 5
    d_model = 16
    n_head = 4

    q = torch.rand(batch_size, seq_length, d_model)
    k = torch.rand(batch_size, seq_length, d_model)
    v = torch.rand(batch_size, seq_length, d_model)

    # 创建多头注意力模块
    mha = MultiHeadAttention(d_model=d_model, n_head=n_head)

    # 计算多头注意力
    out = mha(q, k, v)

    # 打印结果
    print("输出张量形状:", out.size())

