import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    从零开始实现一个多头注意力模块
    """

    def __init__(self, d_model: int, h: int):
        """
        初始化函数
        :param d_model: 模型的嵌入维度 (embedding dimension)
        :param h: 注意力头的数量 (number of heads)
        """
        super(MultiHeadAttention, self).__init__()
        # 确保 d_model 可以被 h 整除
        assert d_model % h == 0, "d_model 必须能被 h 整除"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # 每个头的维度

        # 定义 Q, K, V 和输出的线性层
        # 在实际应用中，可以将 W_q, W_k, W_v 合并为一个大的线性层，然后切分，以提高计算效率
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask=None):
        """
        计算缩放点积注意力
        :param query: Q, shape: (batch, h, seq_len, d_k)
        :param key: K, shape: (batch, h, seq_len, d_k)
        :param value: V, shape: (batch, h, seq_len, d_k)
        :param mask: 掩码, shape: (batch, 1, 1, seq_len)
        :return: 注意力输出和注意力权重
        """
        d_k = query.size(-1)

        # 1. Q, K^T 点积，计算注意力分数
        # (batch, h, seq_len_q, d_k) @ (batch, h, d_k, seq_len_k) -> (batch, h, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. 应用掩码 (Mask)
        if mask is not None:
            # 将掩码中为 0 的位置替换为一个非常小的负数，这样在 softmax 后会趋近于 0
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. 对分数应用 softmax
        attention_weights = F.softmax(scores, dim=-1)

        # 4. 注意力权重与 V 相乘
        # (batch, h, seq_len_q, seq_len_k) @ (batch, h, seq_len_v, d_k) -> (batch, h, seq_len_q, d_k)
        # 注意: seq_len_k == seq_len_v
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播
        :param query: Q, shape: (batch, seq_len_q, d_model)
        :param key: K, shape: (batch, seq_len_k, d_model)
        :param value: V, shape: (batch, seq_len_v, d_model)
        :param mask: 掩码, shape: (batch, 1, seq_len_q, seq_len_k) or (batch, seq_len_q, seq_len_k)
        :return: 最终输出和注意力权重
        """
        batch_size = query.size(0)

        # 1. 线性投影
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # 2. 拆分多头
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        # 这里使用 view 是安全的，因为线性层的输出是连续的
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 如果有掩码，需要调整其形状以适应多头
        if mask is not None:
            # (batch, seq_len_q, seq_len_k) -> (batch, 1, seq_len_q, seq_len_k)
            # 这样可以广播到所有头
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # 3. 计算缩放点积注意力
        # context: (batch, h, seq_len_q, d_k)
        # attention_weights: (batch, h, seq_len_q, seq_len_k)
        context, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 4. 合并多头
        # (batch, h, seq_len_q, d_k) -> (batch, seq_len_q, h, d_k)
        # transpose 操作会使内存不连续，所以后面要用 .contiguous()
        context = context.transpose(1, 2).contiguous()

        # (batch, seq_len_q, h, d_k) -> (batch, seq_len_q, d_model)
        # 使用 .view() 将 h 和 d_k 维度合并回 d_model
        context = context.view(batch_size, -1, self.d_model)

        # 5. 最终线性投影
        # (batch, seq_len_q, d_model) -> (batch, seq_len_q, d_model)
        output = self.w_o(context)

        return output, attention_weights


# --- 使用示例 ---
if __name__ == '__main__':
    # 参数设置
    batch_size = 4
    seq_length = 10
    d_model = 512
    h = 8

    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, h)

    # 创建虚拟输入数据 (在自注意力中, Q, K, V 相同)
    x = torch.randn(batch_size, seq_length, d_model)

    # 创建一个上三角掩码，用于模拟解码器中的情况（防止看到未来的词）
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    mask = mask.unsqueeze(0)  # 添加 batch 维度

    # 前向传播
    output, attn_weights = mha(query=x, key=x, value=x, mask=mask)

    # 打印输出形状
    print(f"输入 x 的形状: {x.shape}")
    print(f"输出 output 的形状: {output.shape}")
    print(f"注意力权重 attn_weights 的形状: {attn_weights.shape}")

    # 验证掩码是否生效
    # 检查注意力权重矩阵的第一行，除了第一个元素外，其他都应接近0
    print("\n检查掩码效果 (第一条数据, 第一个头, 第一个 query token 的注意力权重):")
    print(attn_weights[0, 0, 0, :])