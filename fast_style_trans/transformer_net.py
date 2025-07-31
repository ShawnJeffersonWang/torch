from torch import nn

class ResidualBlock(nn.Module): # 定义残差块
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 第1个卷积层序列
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1), # 边缘反射填充
            nn.Conv2d(channels, channels, kernel_size=3, stride=1), # 卷积层
            nn.InstanceNorm2d(num_features=channels, affine=True), # 实例归一化
            nn.ReLU() # 激活函数
        )
        # 第2个卷积层序列
        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(1), # 边缘反射填充
            nn.Conv2d(channels, channels, kernel_size=3, stride=1), # 卷积层
            nn.InstanceNorm2d(num_features=channels, affine=True) # 实例归一化
        )

    def forward(self, x):
        fx = self.layer1(x) # 输入x先通过layer1
        fx = self.layer2(fx) # 再通过layer2，得到主路径的计算结果
        return x + fx # 返回原始输入x与主路径计算结果fx的和

class TransformerNet(nn.Module): # 定义风格转换网络
    def __init__(self):
        super(TransformerNet, self).__init__()
        # 第1个卷积层序列
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(4), # 边缘反射填充
            nn.Conv2d(3, 32, kernel_size=9, stride=1), # 卷积层
            nn.InstanceNorm2d(num_features=32, affine=True), # 实例归一化
            nn.ReLU() # ReLU激活函数
        )
        # 第2个卷积层序列
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1), # 边缘反射填充
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # 卷积层
            nn.InstanceNorm2d(num_features=64, affine=True), # 实例归一化
            nn.ReLU() # ReLU激活函数
        )
        # 第3个卷积层序列
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1), # 边缘反射填充
            nn.Conv2d(64, 128, kernel_size=3, stride=2), # 卷积层
            nn.InstanceNorm2d(num_features=128, affine=True), # 实例归一化
            nn.ReLU() # ReLU激活函数
        )
        # 定义残差块序列，包括了5个残差块
        self.residuals = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        # 定义上采样序列1
        self.up1 = nn.Sequential(
            # 使用转置卷积层来实现上采样层，同时实现Upsample、ReflectionPad2d和Conv2d的功能
            # 这样可以简化操作、提高计算效率，并有着更好的上采样效果
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_features=64, affine=True), # 实例归一化
            nn.ReLU(), # ReLU激活函数
        )
        # 定义上采样序列2
        self.up2 = nn.Sequential(
            # 使用转置卷积层来实现上采样层，同时实现Upsample、ReflectionPad2d和Conv2d的功能
            # 这样可以简化操作、提高计算效率，并有着更好的上采样效果
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_features=32, affine=True), # 实例归一化
            nn.ReLU(), # ReLU激活函数
        )
        # 定义输出序列
        self.out = nn.Sequential(
            nn.ReflectionPad2d(4),  # 边缘反射填充
            nn.Conv2d(32, 3, kernel_size=9, stride=1)  # 卷积层
        )

    def forward(self, x):
        x = self.conv1(x)  # 经过卷积序列
        x = self.conv2(x)  # 经过卷积序列
        x = self.conv3(x)  # 经过卷积序列
        x = self.residuals(x)  # 经过残差块序列
        x = self.up1(x)  # 经过上采样卷积序列
        x = self.up2(x)  # 经过上采样卷积序列
        x = self.out(x)  # 经过输出序列
        return x

def print_forward(model, x): # 打印前向传播过程的函数
    # 计算过程与forward函数一样，用来观察张量x经过网络后的尺寸变化
    print(f"x.shape: {x.shape}")
    x = model.conv1(x)
    print(f"after conv1: {x.shape}")
    x = model.conv2(x)
    print(f"after conv2: {x.shape}")
    x = model.conv3(x)
    print(f"after conv3: {x.shape}")
    x = model.residuals(x)
    print(f"after residuals: {x.shape}")
    x = model.up1(x)
    print(f"after up1: {x.shape}")
    x = model.up2(x)
    print(f"after up2: {x.shape}")
    x = model.out(x)
    print(f"after out: {x.shape}")
    print("")


import torch

if __name__ == "__main__":
    # 定义5*3*256*256的输入数据x
    x = torch.randn(5, 3, 256, 256)
    model = TransformerNet() # 定义模型
    output = model(x)
    print_forward(model, x) # 打印x的尺寸变化
    print(f"output.shape = {output.shape}") # 输出结果的尺寸

