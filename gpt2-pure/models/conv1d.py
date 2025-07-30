import torch
from torch import nn


# 来自transformers库的pytorch_utils.py中的Conv1D实现
# 具体路径，transformers/src/transformers/pytorch_utils.py

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


if __name__ == '__main__':
    conv = Conv1D(nf=10, nx=5)
    x = torch.randn(2, 8, 5)
    output = conv(x)
    print("output.shape: ", output.shape)
