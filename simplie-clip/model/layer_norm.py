import torch
from torch import nn

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    # 继承 torch.nn.LayerNorm 类，添加对 float16 输入的支持
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype # 保存原始数据类型（可能是 float16）
        # #转成float32执行LayerNorm
        # 这是因为LayerNorm在fp16下可能数值不稳定
        # 然后调用父类的forward方法
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type) # 输出结果再转换回原始数据类型

if __name__ == "__main__":
    norm = LayerNorm(4)
    x_fp32 = torch.randn(2, 4, dtype=torch.float32)
    x_fp16 = x_fp32.half()

    print("Input (float32):")
    print(x_fp32)
    print("Output (float32):")
    print(norm(x_fp32))

    print("\nInput (float16):")
    print(x_fp16)
    print("Output (float16):")
    print(norm(x_fp16))


