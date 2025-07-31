# 用于计算特征图y的格拉姆矩阵
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

# 用于批量归一化
def normalize_batch(batch):
    batch = batch.div_(255.0)
    # 归一化的参数mean和std，是imagenet数据集的参数
    # 因为vgg_loss网络，是基于imagenet数据集训练的
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

from torch.nn.functional import mse_loss
def compute_loss(generate, gram_style, content):
    content_weight = 1e5
    # 根据relu2层的输出，计算内容损失
    content_loss = content_weight * mse_loss(generate['relu2_2'],
                                        content['relu2_2'])
    style_loss = 0.
    for ys, gs in zip(generate.values(), gram_style):
        gy = gram_matrix(ys) # 计算生成图像的格拉姆矩阵
        style_loss += mse_loss(gy, gs) # 计算风格损失
    style_weight = 1e10
    style_loss *= style_weight # 将style_loss乘以系数
    total_loss = content_loss + style_loss # 累加到一起
    return content_loss, style_loss, total_loss

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim

from PIL import Image
from torchvision.utils import save_image
import os

from transformer_net import TransformerNet
from vgg_loss import Vgg16Loss

if __name__ == '__main__':
    # 创建转换器data_transform，对训练数据读取和转换
    data_transform = transforms.Compose([
        transforms.Resize(256), #调整图像大小
        transforms.CenterCrop(256), # 裁剪图像
        transforms.ToTensor(), # 转张量
        transforms.Lambda(lambda x: x.mul(255)) # 缩放至0~255之间的数
    ])
    # 使用ToTensor后，张量会转为0~1之间的数
    # 我们再次将数据缩放回0~255，是因为TransformerNet网络，接收和处理0~255之间的数据
    # 这种情况下，训练效果是最好的

    train_data = datasets.ImageFolder("./data/", data_transform) # 构造数据集
    print(f"traind_data size = {len(train_data)}") # 打印train_data的长度

    batch_size = 4
    train_load = DataLoader(train_data, shuffle = True, batch_size = batch_size)
    # 打印train_load的长度
    print(f"train_loader size = {len(train_load)}")

    # 定义设备对象device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = %s" % (device))

    model = TransformerNet().to(device) #风格转换模型
    optimizer = optim.Adam(model.parameters(), 0.001) #定义Adam优化器
    # 创建计算损失的网络
    vgg_loss = Vgg16Loss('./pretrained/hub/checkpoints/vgg16-397923af.pth').to(device)

    # 定义单张图的转换器
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_image = Image.open("./data/vango.jpg").convert('RGB') # 打开样式图像
    style_image = image_transform(style_image) # 将图像转为张量
    style_image = style_image.repeat(batch_size, 1, 1, 1).to(device) # 复制batch_size个数据
    style_image = normalize_batch(style_image) # 对图像进行标准化
    style = vgg_loss(style_image) # 计算风格图像的特征图
    gram_style = [gram_matrix(s) for s in style.values()] # 计算格拉姆矩阵

    os.makedirs('output', exist_ok=True) # 建立文件夹，保存迭代过程中的测试图片和模型
    print("begin train:")
    model.train() # 将模型调整为训练模式
    n_epoch = 10
    for epoch in range(1, n_epoch + 1): # 进入10轮的循环迭代
        for batch_idx, (content, _) in enumerate(train_load):
            optimizer.zero_grad() # 清空梯度
            content = content.to(device)
            generate = model(content) # 使用转换网络，计算生成图像

            generate = normalize_batch(generate) # 将生成图像归一化
            content = normalize_batch(content) # 将内容图像归一化

            generate = vgg_loss(generate) # 计算生成图像特征图
            content = vgg_loss(content)  # 计算内容图像特征图

            # 计算风格损失、内容损失和总损失
            content_loss, style_loss, total_loss = (
                compute_loss(generate, gram_style, content))

            total_loss.backward() # 计算梯度
            optimizer.step() # 更新模型参数
            
            # 打印调试信息
            print(f"Epoch {epoch}/{n_epoch} " # 迭代轮数
                  f"| Batch {batch_idx + 1}/{len(train_load)} " # 批次编号
                  f"| Content Loss: {content_loss.item():.4f} " # 内容损失
                  f"| Style Loss: {style_loss.item():.4f}") # 风格损失

            # 每训练50个batch保存一次转换效果，用于观察模型的训练情况
            if (batch_idx + 1) % 50 == 0:
                content_path = "./data/tower.jpg" # 测试图像
                test_image = Image.open(content_path).convert('RGB')
                test_image = image_transform(test_image)
                test_image = test_image.unsqueeze(0).to(device)

                model.eval()
                output = model(test_image).cpu()
                # 定义保存路径
                save_path = f"./output/epoch{epoch}_batch{batch_idx + 1}.jpg"
                save_image(output.clamp(0, 255).div(255), save_path) # 保存生成后的图像
                print("Save check image: %s" % (save_path))
                model.train()

        model.eval().cpu()
        save_path = f'./output/vango_{epoch}.pth'
        torch.save(model.state_dict(), save_path) # 保存一次模型
        print("Save model: %s" % (save_path))
        model = model.to(device)
        model.train()


