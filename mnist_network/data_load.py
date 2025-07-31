from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 初学者在学习这一部分时，只要知道大致的数据处理流程就可以了
if __name__ == '__main__':
    # 实现图像的预处理pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图
        transforms.ToTensor()  # 转换为张量
    ])

    # 使用ImageFolder函数，读取数据文件夹，构建数据集dataset
    # 这个函数会将保存数据的文件夹的名字，作为数据的标签，组织数据
    # 例如，对于名字为“3”的文件夹，就会将“3”就会作为文件夹中图像数据的标签，和图像配对，用于后续的训练
    # 使用起来非常的方便
    train_dataset = datasets.ImageFolder(root='./mnist_images/train', transform=transform)
    test_dataset = datasets.ImageFolder(root='./mnist_images/test', transform=transform)
    # 打印它们的长度
    print("train_dataset length: ", len(train_dataset))
    print("test_dataset length: ", len(test_dataset))

    # 使用train_loader，实现小批量的数据读取
    # 这里设置小批量的大小，batch_size=64。也就是每个批次，包括64个数据
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 打印train_loader的长度
    print("train_loader length: ", len(train_loader))
    # 60000个训练数据，如果每个小批量，读入64个样本，那么60000个数据会被分成938组
    # 计算938*64=60032，这说明最后一组，会不够64个数据

    # 循环遍历train_loader
    # 每一次循环，都会取出64个图像数据，作为一个小批量batch
    for batch_idx, (data, label) in enumerate(train_loader):
        if batch_idx == 3: # 打印前3个batch观察
            break
        print("batch_idx: ", batch_idx)
        print("data.shape: ", data.shape) # 数据的尺寸
        print("label: ", label.shape) # 图像中的数字
        print(label)























