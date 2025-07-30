import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_
from torch.nn.init import zeros_
import math

from attention import MultiHeadAttention
from mlp import MLP


class Block(nn.Module):
    def __init__(self, n_embd, n_head, drop_rate):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, drop_rate)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd * 4, n_embd, drop_rate)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, drop_rate, n_layer):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(drop_rate)
        self.h = nn.ModuleList([Block(n_embd, n_head, drop_rate)
                                for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx, pos):
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, drop_rate, n_layer):
        super().__init__()
        self.transformer = Transformer(vocab_size, block_size, n_embd, n_head, drop_rate, n_layer)
        self.block_size = block_size
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer(idx, pos)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    # 设置模型参数
    vocab_size = 10000  # 词汇表大小
    block_size = 512  # 序列最大长度
    n_embd = 768  # 嵌入层大小
    n_head = 12  # 注意力机制的头数
    drop_rate = 0.1  # Dropout比率
    n_layer = 6  # Transformer块的数量

    # 初始化模型
    model = GPT2(vocab_size, block_size, n_embd, n_head, drop_rate, n_layer)
    model.eval()  # 将模型设置为评估模式

    # 创建虚拟输入数据
    batch_size = 2
    sequence_length = 50  # 小于block_size以确保不出错
    dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_length))

    # 测试前向传播（推理）
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)  # 应该是 [batch_size, vocab_size]

    # 测试参数初始化是否正常
    for name, param in model.named_parameters():
        print(f"{name}: mean={param.mean().item()}, std={param.std().item()}")  # 输出参数的统计信息

    # 某种初始化的序列索引，通常是编码过的
    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    # 生成10个新的词
    generated_idx = model.generate(idx, max_new_tokens=10)
    print(generated_idx)
