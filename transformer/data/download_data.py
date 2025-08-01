# 需要安装 datasets 库
# 这是由 Hugging Face 提供的一个用于下载、缓存、保存、加载和预处理常见数据集的库
# conda install -c huggingface datasets
# https://huggingface.co/datasets/iwslt2017/resolve/main/data/2017-01-trnted/texts/en/zh/en-zh.zip

import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

from datasets import load_dataset

download_path = "./raw_data"
import os
os.makedirs(download_path, exist_ok = True)

print("download data...")
# 通过load_dataset，下载数据集，下载后，会得到数据集对象dataset
dataset = load_dataset("iwslt2017", "iwslt2017-en-zh", cache_dir=download_path)

print("dataset path:")
for name in dataset.cache_files:
    # 遍历cache_files，可以打印数据的保存目录
    print("%s %s\n"%(name, dataset.cache_files[name]))

# 计算train和test包含的数据数量
train_num = len(dataset['train'])
test_num = len(dataset['test'])
print("")
# 将它们打印出来
print("train data len: %d"%(train_num))
print("test data len: %d"%(test_num))

