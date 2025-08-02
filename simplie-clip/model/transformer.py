import torch
from torch import nn

from model.layer_norm import LayerNorm
from model.block import ResidualAttentionBlock

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,
                       attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int):

        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # shape = [*, width, grid, grid]
        x = self.conv1(x)
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # shape = [*, grid ** 2, width]
        x = x.permute(0, 2, 1)
        # shape = [*, grid ** 2 + 1, width]
        zero = torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        class_token = self.class_embedding.to(x.dtype) + zero

        x = torch.cat([class_token, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

if __name__ == "__main__":
    # 假设输入图像大小为 224x224，patch_size 为 16
    input_resolution = 224
    patch_size = 16
    width = 768
    layers = 12
    heads = 12
    output_dim = 512

    model = VisionTransformer(
        input_resolution=input_resolution,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        output_dim=output_dim
    )

    dummy_input = torch.randn(1, 3, input_resolution, input_resolution)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应该为 [1, output_dim]


