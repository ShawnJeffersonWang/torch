from torch import nn
import math
from conv1d import Conv1D


# 来自transformers库的activations.py中的NewGELUActivation实现
class NewGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# 来自transformers库的modeling_gpt2.py中的GPT2MLP实现
class MLP(nn.Module):
    def __init__(self, intermediate_size, embed_dim, resid_pdrop):
        super().__init__()
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


import torch

if __name__ == '__main__':
    embed_dim = 128
    intermediate_size = embed_dim * 4
    resid_pdrop = 0.1
    mlp = MLP(intermediate_size, embed_dim, resid_pdrop)
    x = torch.rand(10, embed_dim)
    output = mlp(x)
    print("x:", x.shape)
    print("output.shape:", output.shape)
