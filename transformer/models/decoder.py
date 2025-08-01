from torch import nn
from models.attention import MultiHeadAttention
from models.feed_forward import FeedForward
from models.positional_encoding import PositionalEncoding

class DecoderLayer(nn.Module):
    # 初始化init函数
    # 参数d_model：Transformer特征维度
    # 参数ffn_hidden：前馈神经网络隐藏层维度
    # 参数n_head：多头注意力的头数
    # 参数drop_prob：丢弃比率
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        # 定义第一个注意力机制，用于对解码器的输入数据dec进行编码
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(drop_prob) # dropout层
        self.norm1 = nn.LayerNorm(d_model) # 层归一化

        # 定义第2个注意力机制，用于结合编码器的输出enc和解码器的输入dec，进行计算
        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.dropout2 = nn.Dropout(drop_prob) # dropout层
        self.norm2 = nn.LayerNorm(d_model) # 层归一化

        # 前馈神经网络
        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob)
        self.dropout3 = nn.Dropout(drop_prob) # dropout层
        self.norm3 = nn.LayerNorm(d_model) # 层归一化

    # 模型的推理函数，传入解码器输入dec、编码器输出enc
    # dec和enc对应的掩码数组trg_mask和src_mask
    def forward(self, dec, enc, trg_mask, src_mask):
        # 1.带有掩码的自注意力计算
        residual = dec # 保存残差
        # 将dec，作为q、k、v，与trg_mask，一起传入self_attention计算
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout1(x) # 计算dropout层
        x = self.norm1(x + residual) # 计算层归一化

        # 2.编码器、解码器注意力计算
        residual = x # 保存残差
        # 将上一个多头注意力的输出x作为q，编码器的输出enc作为k和v
        # 与src_mask一起传入enc_dec_attention计算
        x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
        x = self.dropout2(x) # 计算dropout层
        x = self.norm2(x + residual) # 计算层归一化

        # 3.前馈神经网络的计算
        residual = x # 保存残差
        x = self.ffn(x) # 使用ffn网络计算模型结果
        x = self.dropout3(x) # 计算dropout层
        x = self.norm3(x + residual) # 计算层归一化
        return x # 返回计算后的x


# 解码器模块Decoder
class Decoder(nn.Module):
    # 初始化函数
    def __init__(self, trg_vocab_size, max_len,
                 d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        self.embed = nn.Embedding(trg_vocab_size, d_model) # 词嵌入层
        self.position = PositionalEncoding(d_model, max_len) # 位置编码层
        self.dropout = nn.Dropout(drop_prob) # dropout层

        # 解码器层
        self.layers = nn.ModuleList([
                      DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                      for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, trg_vocab_size) # 用于计算预测结果

    # 推理计算函数，函数传入解码器的输入x，编码器的输出enc_src
    # trg_mask和src_mask，两个掩码数组
    def forward(self, x, enc_src, trg_mask, src_mask):
        x = self.embed(x) # 词嵌入层
        x = self.position(x) # 位置编码层
        x = self.dropout(x) # dropout层
        for layer in self.layers:
            # x和enc_src，一起输入至n个串联的解码器层
            x = layer(x, enc_src, trg_mask, src_mask)
        x = self.linear(x) # 将x输入至linear层，计算预测结果
        return x # 返回
















import torch

if __name__ == "__main__":
    # 定义模型参数
    trg_vocab_size = 1000  # 目标词汇表大小为1000
    max_len = 50  # 最大序列长度
    d_model = 512  # Transformer特征维度
    ffn_hidden = 2048  # 前馈神经网络隐藏层维度
    n_head = 8  # 多头注意力的头数
    n_layers = 6  # 编码器层数
    drop_prob = 0.1  # 丢弃比率

    # 解码器实例
    model = Decoder(trg_vocab_size = trg_vocab_size,
                      max_len = max_len,
                      d_model = d_model,
                      ffn_hidden = ffn_hidden,
                      n_head = n_head,
                      n_layers = n_layers,
                      drop_prob = drop_prob)

    # 解码器的输入序列
    trg = torch.randint(0, trg_vocab_size, (5, 12))
    enc = torch.randn(5, 15, d_model)  # 编码器的输出向量
    # 两个掩码数组
    trg_mask = torch.ones(5, 1, 1, 12)
    src_mask = torch.ones(5, 1, 1, 15)

    # 将它们一起输入至model，计算解码结果
    output = model(trg, enc, trg_mask, src_mask)

    # 打印output的尺寸，会看到output是一个5×12×1000的张量
    print(output.shape)


