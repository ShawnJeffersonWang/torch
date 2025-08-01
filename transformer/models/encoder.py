from torch import nn
from models.attention import MultiHeadAttention
from models.feed_forward import FeedForward
from models.positional_encoding import PositionalEncoding

class EncoderLayer(nn.Module):
    # 初始化init函数
    # 参数d_model：Transformer特征维度
    # 参数ffn_hidden：前馈神经网络隐藏层维度
    # 参数n_head：多头注意力的头数
    # 参数drop_prob：丢弃比率
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        # 橙色的多头注意力机制
        self.attention = MultiHeadAttention(d_model, n_head)
        self.drop1 = nn.Dropout(drop_prob) # dropout层
        self.norm1 = nn.LayerNorm(d_model) # 层归一化
        # 蓝色的前馈神经网络
        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob)
        self.drop2 = nn.Dropout(drop_prob) # dropout层
        self.norm2 = nn.LayerNorm(d_model) # 层归一化
    # 模型的推理函数
    # 传入输入张量x和x对应的掩码数组src_mask
    def forward(self, x, src_mask):
        # 计算步骤1：多头注意力机制计算
        residual = x # 保存残差
        # 将x，作为q、k、v，与src_mask，一起传入attention
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        # 将计算后的x输入至drop1
        x = self.drop1(x)
        # 将x和residual相加，一起输入到层归一化
        x = self.norm1(x + residual)
        # 计算步骤2：前馈神经网络的计算
        residual = x # 保存残差
        x = self.ffn(x) # 计算模型结果
        x = self.drop2(x) # dropout层
        x = self.norm2(x + residual) # 层归一化
        return x # 返回计算后的x

class Encoder(nn.Module):
    # 初始化函数
    def __init__(self, src_vocab_size, max_len,
                 d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, d_model) # 词嵌入层
        self.position = PositionalEncoding(d_model, max_len) # 位置编码层
        self.dropout = nn.Dropout(drop_prob) # dropout层
        self.layers = nn.ModuleList() # 编码器层

        for _ in range(n_layers):
            # 使用循环，构造n个编码器层
            encoder_layer = EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
            # 将这n个编码器层，保存在layers中
            self.layers.append(encoder_layer)

    # 推理计算函数
    # 函数传入输入数据x，和x的掩码数组src_mask
    def forward(self, x, src_mask):
        x = self.embed(x) # 词嵌入层
        x = self.position(x) # 位置编码层
        x = self.dropout(x) # dropout层
        for layer in self.layers:
            # 接着x会输入至n个串联的编码器层
            x = layer(x, src_mask)
        return x # 返回编码后的张量x


# 编写一个print_forward函数
# 用来打印encoder推理计算时，张量的尺寸变化
def print_forward(model, x):
    # 输入的张量x是5×12
    print("encoder forward:")
    print(f"input x: {x.shape}")
    x = model.embed(x)
    # 经过embed后，尺寸变为5×12×512
    print(f"after embed: {x.shape}")
    x = model.position(x)
    print(f"after position: {x.shape}")
    x = model.dropout(x)
    print(f"after dropout: {x.shape}")

    for layer in model.layers:
        x = layer(x, src_mask)
    # 最终输出仍然是5×12×512
    print(f"after layers: {x.shape}")
    print("")


import torch
# 测试程序
if __name__ == "__main__":
    # 定义模型参数
    src_vocab_size = 1000  # 源词汇表大小为1000
    max_len = 50  # 最大序列长度
    d_model = 512  # Transformer特征维度
    ffn_hidden = 2048  # 前馈神经网络隐藏层维度
    n_head = 8  # 多头注意力的头数
    n_layers = 6  # 编码器层数
    drop_prob = 0.1  # 丢弃比率

    # 创建编码器实例
    model = Encoder(src_vocab_size = src_vocab_size,
                    max_len = max_len,
                    d_model = d_model,
                    ffn_hidden = ffn_hidden,
                    n_head = n_head,
                    n_layers = n_layers,
                    drop_prob = drop_prob)

    # 输入序列x，它是一个5×12的张量，表示5个数据，每个数据包括12个单词
    x = torch.randint(0, src_vocab_size, (5, 12))
    # x对应的掩码数组
    src_mask = torch.ones(5, 1, 1, 12)

    # 用来打印encoder推理计算时，张量的尺寸变化
    print_forward(model, x)

    # 将x输入至model，计算编码结果
    output = model(x, src_mask)
    # 输出结果
    print("Output shape:", output.shape)

