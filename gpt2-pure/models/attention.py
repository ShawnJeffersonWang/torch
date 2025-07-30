import torch
from torch import nn
import math
from conv1d import Conv1D


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(query, key, value, dropout_p):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, drop_rate):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = Conv1D(3 * n_embd, n_embd)
        self.c_proj = Conv1D(n_embd, n_embd)
        self.dropout = nn.Dropout(drop_rate)
        self.n_head = n_head
        self.n_embd = n_embd
        self.drop_rate = drop_rate

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        dropout_p = self.drop_rate if self.training else 0
        y = scaled_dot_product_attention(q, k, v, dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.dropout(y)
        return y


if __name__ == '__main__':
    # Define dimensions and create a dummy input
    n_embd = 128
    n_head = 8
    drop_rate = 0.1
    batch_size, seq_length = 2, 60

    # Initialize model and data
    model = MultiHeadAttention(n_embd, n_head, drop_rate)
    x = torch.randn(batch_size, seq_length, n_embd)

    # Forward pass
    out = model(x)
    print("Output shape:", out.shape)
