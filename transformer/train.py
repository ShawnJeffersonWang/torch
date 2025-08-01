import torch
from torch import nn

from models.transformer import Transformer
from dataset import split_token
from dataset import TranslateDataset
from dataset import build_vocab
from dataset import collate_batch
from translator import Translator

# 函数传入模型model和src_vocab与trg_vocab两个词表
def test_translate(model, src_vocab, trg_vocab):
    sample = "<sos> I like math , you like english . <eos>"  # 定义一个测试样本
    src_tokens = split_token(sample)  # 分词结果
    src_index = [src_vocab[token] for token in src_tokens]  # 通过词表转为词语的索引
    src_tensor = torch.LongTensor(src_index).unsqueeze(0).to(DEVICE)  # 转为张量

    beam_size = 5
    max_seq_len = 16
    src_pad_idx = src_vocab["<pad>"]
    trg_pad_idx = trg_vocab["<pad>"]
    trg_bos_idx = trg_vocab["<sos>"]
    trg_eos_idx = trg_vocab["<eos>"]

    translator = Translator(model,
                            beam_size,
                            max_seq_len,
                            src_pad_idx,
                            trg_pad_idx,
                            trg_bos_idx,
                            trg_eos_idx).to(DEVICE)

    translated_indexes = translator.translate_sentence(src_tensor)

    trg_itos = trg_vocab.get_itos()

    predict_word = [trg_itos[index] for index in translated_indexes]
    print("I like math . -> ", end="")
    print("".join(predict_word))  # 打印出来


import pickle
from torch.utils.data import DataLoader
from torch import optim
import os

if __name__ == '__main__':
    # 使用TranslateDataset读取训练数据，得到数据集dataset
    dataset = TranslateDataset("./data/small.txt")
    #dataset = TranslateDataset("./data/train.txt")

    # 使用build_vocab，建立源语言词表和目标语言词表
    src_vocab, trg_vocab = build_vocab(dataset)
    # 打印两个词表长度
    print("src_vocab:", len(src_vocab))
    print("trg_vocab:", len(trg_vocab))

    os.makedirs('output', exist_ok=True)  # 建立文件夹，保存迭代过程中的测试图片和模型

    # 将两个词表保存下来，词表也相当于模型的一部分
    with open("output/src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open("output/trg_vocab.pkl", "wb") as f:
        pickle.dump(trg_vocab, f)

    # 定义一个符合DataLoader中collate_fn参数形式的函数collate
    collate = lambda batch: collate_batch(batch, src_vocab, trg_vocab)
    # 定义dataloader读取dataset
    dataloader = DataLoader(dataset,
                            batch_size=4,  # 每个小批量包含4个数据
                            shuffle=True,  # 将数据打乱顺序后读取
                            collate_fn=collate)

    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print("DEVICE = ", DEVICE)

    # 定义模型的必要参数
    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)
    src_pad_idx = src_vocab["<pad>"]
    trg_pad_idx = trg_vocab["<pad>"]

    #model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(DEVICE)

    # 初始化Transformer模型
    model = Transformer(src_vocab_size=src_vocab_size, src_pad_idx=src_pad_idx,
                              trg_vocab_size=trg_vocab_size, trg_pad_idx=trg_pad_idx,
                              max_len=256, d_model=512, n_head=8,
                              ffn_hidden=2048, n_layers=8, drop_prob=0.1).to(DEVICE)

    model.train()  # 将model调整为训练模式

    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 定义Adam优化器
    # 定义交叉熵损失函数，需要将<pad>标签设置为ignore_index
    criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)

    print("begin train:")
    n_epoch = 200  # 训练轮数设置为200
    for epoch in range(1, n_epoch + 1):  # 外层循环，代表了整个训练数据集的遍历次数
        # 内层循环代表了，在一个epoch中
        # 以随机梯度下降的方式，使用dataloader对于数据进行遍历
        # batch_idx表示当前遍历的批次
        # (text, pos_tag) 表示这个批次的训练数据和词性标记
        for batch_idx, (src, trg) in enumerate(dataloader):  # 遍历dataloader
            # 将src和trg移动到当前设备DEVICE上
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)

            optimizer.zero_grad()  # 将梯度清零

            # 使用模型model，计算预测结果predict
            predict = model(src, trg[:, :-1])  # 使用模型model计算text的预测结果
            #predict = model(src, trg)

            # 使用view调整predict和标签trg的维度
            predict = predict.view(-1, predict.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)
            #trg = trg.contiguous().view(-1)

            loss = criterion(predict, trg)  # 计算损失
            loss.backward()  # 计算损失函数关于模型参数的梯度
            # 裁剪梯度，防止梯度爆炸
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()  # 更新模型参数

            # 打印调试信息。包括了当前的迭代轮数epoch
            # 当前的批次batch
            # 当前这个批次的损失loss.item
            print(f"Epoch {epoch}/{n_epoch} "
                  f"| Batch {batch_idx + 1}/{len(dataloader)} "
                  f"| Loss: {loss.item():.4f}")
            # 打印某一个固定样本的翻译效果，观察翻译效果的变化
            
            if (batch_idx + 1) % 20 == 0:
                test_translate(model, src_vocab, trg_vocab)

        test_translate(model, src_vocab, trg_vocab)

        """
        model.eval().cpu()
        save_path = f'./output/en2zh_{epoch}.pth'
        torch.save(model.state_dict(), save_path)  # 保存一次模型
        print("Save model: %s" % (save_path))
        model = model.to(DEVICE)
        model.train()
        """

        # 将训练好的模型保存为文件，文件名为translate.model
    torch.save(model.state_dict(), 'output/en2zh.pth')

