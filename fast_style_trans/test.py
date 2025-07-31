import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from transformer_net import TransformerNet

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = %s" % (device))
    # 转换工具，转换任意图像
    content_image = Image.open("./data/airplane.jpg").convert('RGB')
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    model = TransformerNet()
    state_dict = torch.load("vango.pth") # 读入训练好的vango.pth模型
    model.load_state_dict(state_dict) # 使用model，转换任意的输入图像
    model.to(device)
    model.eval()

    output = model(content_image).cpu()
    save_image(output.clamp(0, 255).div(255), "./data/airplane_vango.jpg")



