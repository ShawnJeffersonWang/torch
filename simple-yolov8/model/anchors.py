import torch

def make_anchors(x, strides, offset):
    anchor_points = list()
    stride_tensor = list()
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

if __name__ == "__main__":
    # 创建模拟的特征图张量
    feature_maps = [
        torch.randn(1, 256, 10, 10),
        torch.randn(1, 256, 20, 20)
    ]
    strides = [8, 16]  # 假设对于不同尺寸的特征图使用不同的步长
    offset = 0.5  # 锚点位于像素中心

    # 调用函数
    anchors, anchor_strides = make_anchors(feature_maps, strides, offset)

    # 打印输出结果
    print("Anchors:\n", anchors.shape)
    print("Anchor Strides:\n", anchor_strides.shape)
