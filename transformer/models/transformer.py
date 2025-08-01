import torch
from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder


# 计算源语言序列src的掩码数组
def get_src_mask(seq, pad_idx):
    pad_mask = (seq != pad_idx) # 填充掩码
    return pad_mask.unsqueeze(1).unsqueeze(2)

    # 计算源语言序列trg的掩码数组
def get_trg_mask(seq, pad_idx):
    # 创建填充掩码
    trg_pad_mask = (seq != pad_idx).unsqueeze(1).unsqueeze(3)
    seq_len = seq.shape[1]
    ones = torch.ones(seq_len, seq_len, device=seq.device)
    # 创建前瞻掩码
    trg_sub_mask = torch.tril(ones).bool()
    return trg_pad_mask & trg_sub_mask

class Transformer(nn.Module):
    # Transformer模型的定义
    def __init__(self,  # 参数1到参数5，对应了是输入数据的情况
                 src_vocab_size,  # 参数1: 源语言英文词汇表大小
                 src_pad_idx,  # 参数2: 源语言填充单词索引
                 trg_vocab_size,  # 参数3: 目标语言中文词汇表大小
                 trg_pad_idx,  # 参数4: 目标语言填充单词索引
                 max_len,  # 参数5: 输入序列的最大长度

                 # 参数6到参数10，是有关模型尺寸的参数
                 d_model,  # 参数6: Transformer模型内部组件的维度
                 n_head,  # 参数7: 多头注意力的头数
                 ffn_hidden,  # 参数8: 前馈神经网络的隐藏层维度
                 n_layers,  # 参数9: 编码器、解码器内部的层数
                 drop_prob):  # 参数10: dropout率
        super().__init__()
        # 定义编码器
        self.encoder = Encoder(src_vocab_size=src_vocab_size,
                               max_len=max_len,
                               # 参数6到参数10，5个有关模型尺寸的参数
                               d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               n_layers=n_layers,
                               drop_prob=drop_prob)
        # 定义解码器
        self.decoder = Decoder(trg_vocab_size=trg_vocab_size,
                               max_len=max_len,
                               # 参数6到参数10，5个有关模型尺寸的参数
                               d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               n_layers=n_layers,
                               drop_prob=drop_prob)
        # 输入数据中的填充索引，这个索引会用来生成掩码数组
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    # Transformer的推理计算
    def forward(self, src, trg):
        # 计算源语言序列src的掩码数组
        src_mask = get_src_mask(src, self.src_pad_idx)
        # 计算源语言序列trg的掩码数组
        trg_mask = get_trg_mask(trg, self.trg_pad_idx)
        # 计算Inputs输入序列的编码结果
        enc_src = self.encoder(src, src_mask)
        # 计算最终的预测结果
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output # 返回

if __name__ == "__main__":
    # 定义10个参数
    src_vocab_size = 1234 # 参数1: 源语言英文词汇表大小
    src_pad_idx = 0 # 参数2: 源语言填充单词索引
    trg_vocab_size = 5678 # 参数3: 目标语言中文词汇表大小
    trg_pad_idx = 0 # 参数4: 目标语言填充单词索引
    max_len = 500 # 参数5: 输入序列的最大长度
    d_model = 512 # 参数6: Transformer模型内部组件的维度
    n_head = 8 # 参数7: 多头注意力的头数
    ffn_hidden = 2048 # 参数8: 前馈神经网络的隐藏层维度
    n_layers = 3 # 参数9: 编码器、解码器内部的层数
    drop_prob = 0.1 # 参数10: dropout率

    # 初始化Transformer模型，初始化一个模型实例
    model = Transformer(src_vocab_size=src_vocab_size,
                        src_pad_idx=src_pad_idx,
                        trg_vocab_size=trg_vocab_size,
                        trg_pad_idx=trg_pad_idx,
                        max_len=max_len,
                        d_model=d_model,
                        n_head=n_head,
                        ffn_hidden=ffn_hidden,
                        n_layers=n_layers,
                        drop_prob=drop_prob)

    batch_size = 5 # 批量大小
    src_len = 123 # src的序列长度
    trg_len = 456 # trg的序列长度
    # 生成数值在单词表大小之内，尺寸为batch_size×序列长度的张量
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    trg = torch.randint(0, trg_vocab_size, (batch_size, trg_len))
    # 将src和trg传入模型model，进行推理，计算出推理结果
    output = model(src, trg)
    # 打印src、trg和output的尺寸
    print("src.shape: ", src.shape)
    print("trg.shape: ", trg.shape)
    print("output.shape: ", output.shape)

