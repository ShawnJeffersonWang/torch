from torch import nn
import torch
from model.conv import Conv
from model.dfl import DFL
from model.anchors import make_anchors
import math

def inference_output(x, stride, nc, ch, dfl):
    # 1. 生成锚点和步长信息
    anchors_and_strides = make_anchors(x, stride, 0.5)  # 0.5是网格偏移量
    anchors = anchors_and_strides[0].transpose(0, 1)  # 转置锚点坐标
    strides = anchors_and_strides[1].transpose(0, 1)  # 转置步长信息

    # 2. 将三个特征图的预测结果展平并拼接在一起
    batch_size = x[0].shape[0]  # 获取批量大小
    processed_outputs = []
    no = nc + ch * 4  # 每个锚点的输出维度(类别数+4个坐标*每个坐标的通道数)
    for feature_map in x:
        flattened = feature_map.view(batch_size, no, -1)  # 展平特征图
        processed_outputs.append(flattened)
    x = torch.cat(processed_outputs, dim=2)  # 沿锚点维度拼接

    # 3. 分割预测结果为box坐标和类别分数两部分
    box_channels = ch * 4  # 边界框预测的通道数
    box_pred = x[:, :box_channels, :]  # 提取边界框预测部分
    cls_pred = x[:, box_channels:, :]  # 提取类别预测部分

    # 4. 处理box预测结果(使用DFL分布焦点损失)
    dfl_output = dfl(box_pred)  # 应用DFL模块
    a, b = torch.split(dfl_output, 2, dim=1)  # 分割为左上和右下偏移量

    # 5. 计算最终box坐标
    anchors = anchors.unsqueeze(0)  # 增加批次维度
    left = anchors - a  # 计算左上坐标
    right = anchors + b  # 计算右下坐标
    center = (left + right) / 2  # 计算中心点
    width_height = right - left  # 计算宽高
    final_box = torch.cat((center, width_height), dim=1)  # 合并为中心+宽高格式

    # 6. 调整box坐标到原图尺度并处理类别预测
    scaled_box = final_box * strides  # 根据步长缩放回原图尺寸
    cls_scores = cls_pred.sigmoid()  # 对类别分数应用sigmoid激活

    # 7. 合并最终结果 [B, 4+nc, num_anchors]
    out = torch.cat((scaled_box, cls_scores), dim=1)  # 合并边界框和类别预测
    return out


class Head(nn.Module):
    def __init__(self, nc, filters):
        super().__init__()
        self.nc = nc  # 目标检测的类别数量
        self.ch = 16  # DFL通道数(每个坐标用16个离散bin表示)
        # 每个边界框的坐标(x, y, w, h)各用16维表示

        # 计算cls和box分支的通道数
        c1 = max(filters[0], self.nc)  # cls分支最小通道数
        c2 = max(filters[0] // 4, self.ch * 4)  # box分支最小通道数

        # 初始化cls和box分支的模块列表
        self.cls = nn.ModuleList()  # 类别预测分支
        self.box = nn.ModuleList()  # 边界框预测分支

        # 为每个输入特征图创建对应的预测分支
        for x in filters:
            # 类别预测分支: 两个Conv层+一个1x1卷积
            cls_seq = nn.Sequential(
                Conv(x, c1, 3, 1),  # 3x3卷积
                Conv(c1, c1, 3, 1),  # 3x3卷积
                nn.Conv2d(c1, self.nc, 1)  # 1x1卷积输出类别预测
            )
            self.cls.append(cls_seq)

            # 边界框预测分支: 两个Conv层+一个1x1卷积
            box_seq = nn.Sequential(
                Conv(x, c2, 3, 1),  # 3x3卷积
                Conv(c2, c2, 3, 1),  # 3x3卷积
                # 1x1卷积输出边界框预测(4坐标*每坐标ch通道)
                nn.Conv2d(c2, 4 * self.ch, 1)  
            )
            self.box.append(box_seq)

        self.dfl = DFL(self.ch)  # 初始化DFL模块
        # 初始化一个与检测层数量相同长度的零向量，用于存储各层的步长
        self.stride = torch.zeros(len(filters))

    def forward(self, x):
        # 使用更具描述性的变量名p3、p4、p5代替x[0]、x[1]、x[2]
        p3, p4, p5 = x  # 解包三个特征图(P3,P4,P5)

        # 处理每个特征图，分别通过box和cls分支并拼接结果
        p3 = torch.cat((self.box[0](p3), self.cls[0](p3)), 1)  # 处理P3特征图
        p4 = torch.cat((self.box[1](p4), self.cls[1](p4)), 1)  # 处理P4特征图
        p5 = torch.cat((self.box[2](p5), self.cls[2](p5)), 1)  # 处理P5特征图

        # 将处理后的特征图重新组合成列表
        x = [p3, p4, p5]
        if self.training:  # 训练模式直接返回特征图
            return x
        # 推理模式调用inference_output处理输出
        out = inference_output(x, self.stride, self.nc, self.ch, self.dfl)
        return out

    def initialize_biases(self):
        """初始化偏置项"""
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # 初始化box分支偏置
            # 初始化cls分支偏置(.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


if __name__ == "__main__":
    # 测试Head模块
    # 创建Head对象(类别数80, 输入通道数[64,128,256])
    head = Head(nc=80, filters=[64, 128, 256])

    # 生成模拟输入数据(三个不同尺度的特征图)
    x = [torch.rand(1, 64, 80, 80),  # P3特征图
         torch.rand(1, 128, 40, 40),  # P4特征图
         torch.rand(1, 256, 20, 20)]  # P5特征图

    # 训练模式测试
    outputs = head(x)
    # 打印训练模式下的输出维度
    for output in outputs:
        print("训练模式输出维度:", output.shape)

    # 切换到评估模式
    head.eval()
    # 推理模式测试
    outputs = head(x)
    # 打印推理模式下的输出维度
    print("推理模式输出维度:", outputs.shape)

