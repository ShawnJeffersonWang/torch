import torch
from torch import nn
from collections import OrderedDict
from model.layer_norm import LayerNorm

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor):
        # Step 1: 前向 LayerNorm（用于 attention）
        ln1_out = self.ln_1(x)
        # Step 2: 注意力计算
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        attn_out = self.attn(ln1_out, ln1_out, ln1_out,
                             attn_mask=self.attn_mask)[0]
        # Step 3: 残差连接（attention 部分）
        x = x + attn_out
        # Step 4: 第二个 LayerNorm（用于 MLP）
        ln2_out = self.ln_2(x)
        # Step 5: 前馈神经网络（MLP）
        mlp_out = self.mlp(ln2_out)
        # Step 6: 残差连接（MLP 部分）
        x = x + mlp_out
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    d_model = 8
    n_head = 2
    seq_len = 4
    batch_size = 2
    # 创建一个注意力掩码（可选，也可以为 None）
    attn_mask = torch.zeros(seq_len, seq_len)
    # 实例化模块
    block = ResidualAttentionBlock(d_model=d_model, n_head=n_head, attn_mask=attn_mask)
    # 输入张量（注意 shape: [seq_len, batch_size, d_model]）
    x = torch.randn(seq_len, batch_size, d_model)
    # 前向传播
    out = block(x)
    # 打印结果
    print("Input x:")
    print(x)
    print("\nOutput x:")
    print(out)


