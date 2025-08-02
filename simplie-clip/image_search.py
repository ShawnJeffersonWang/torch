import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from load_model import load
from tokenizer import tokenize

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载CLIP模型
model, preprocess = load("./ViT-B-32.pt", device=device)

# 图片文件夹路径
image_folder = "./data"  # 替换为你的图片文件夹

# 加载图片
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
images = [Image.open(os.path.join(image_folder, f)) for f in image_files]

# 文本查询
text_query = "a dog on the grass"  # 替换为你想要的描述

# 处理文本
text_input = tokenize([text_query]).to(device)

# 处理图片
image_inputs = torch.cat([preprocess(img).unsqueeze(0) for img in images]).to(device)

text_features = model.encode_text(text_input)
image_features = model.encode_image(image_inputs)

# 计算相似度
similarity = text_features @ image_features.T  # 形状: [1, 图片数量]
scores = similarity[0]

# 获取最高分的3张图片
top_scores, top_indices = torch.topk(scores, k=3)

# 显示结果
plt.figure(figsize=(15, 5))
for i, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
    plt.subplot(1, 3, i)
    plt.imshow(images[idx])
    plt.title(f"Score: {score:.2f}")
    plt.axis('off')
plt.show()
