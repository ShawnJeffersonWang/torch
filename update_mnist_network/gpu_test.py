from model import Network
from torchvision import transforms
from torchvision import datasets
import torch

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    # 读取测试数据集
    test_dataset = datasets.ImageFolder(root='./mnist_images/test', transform=transform)
    print("test_dataset length: ", len(test_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    model = Network().to(device)  # 模型转移到GPU
    model.load_state_dict(torch.load('mnist_rotate.pth', map_location=device))

    right = 0 # 保存正确识别的数量
    for i, (x, y) in enumerate(test_dataset):
        # 数据转移到GPU
        x = x.to(device)

        output = model(x)  # 将其中的数据x输入到模型
        predict = output.argmax(1).item() # 选择概率最大标签的作为预测结果
        # 对比预测值predict和真实标签y
        if predict == y:
            right += 1
        else:
            # 将识别错误的样例打印了出来
            img_path = test_dataset.samples[i][0]
            print(f"wrong case: predict = {predict} y = {y} img_path = {img_path}")

    # 计算出测试效果
    sample_num = len(test_dataset)
    acc = right * 1.0 / sample_num
    print("test accuracy = %d / %d = %.3lf" % (right, sample_num, acc))
