import pickle
import torch

from models.transformer import Transformer
from translator import Translator
from dataset import split_token

if __name__ == '__main__':
    # 读入src_vocab和trg_vocab两个词汇表
    with open("./output/src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("./output/trg_vocab.pkl", "rb") as f:
        trg_vocab = pickle.load(f)

    # 打印两个词表长度
    print("text_vocab:", len(src_vocab))
    print("pos_vocab:", len(trg_vocab))

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)
    src_pad_idx = src_vocab["<pad>"]
    trg_pad_idx = trg_vocab["<pad>"]
    trg_bos_idx = trg_vocab["<sos>"]
    trg_eos_idx = trg_vocab["<eos>"]

    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print("DEVICE = ", DEVICE)

    model = Transformer(src_vocab_size=src_vocab_size, src_pad_idx=src_pad_idx,
                              trg_vocab_size=trg_vocab_size, trg_pad_idx=trg_pad_idx,
                              max_len=256, d_model=512, n_head=8,
                              ffn_hidden=2048, n_layers=8, drop_prob=0.1).to(DEVICE)

    model.load_state_dict(torch.load('./output/en2zh.pth'))
    model.eval()

    beam_size = 5
    max_seq_len = 16

    translator = Translator(model,
                            beam_size,
                            max_seq_len,
                            src_pad_idx,
                            trg_pad_idx,
                            trg_bos_idx,
                            trg_eos_idx).to(DEVICE)

    sample = "<sos> I am a student . <eos>"  # 定义一个测试样本
    src_tokens = split_token(sample)  # 分词结果
    src_index = [src_vocab[token] for token in src_tokens]  # 通过词表转为词语的索引
    src_tensor = torch.LongTensor(src_index).unsqueeze(0).to(DEVICE)  # 转为张量

    translated_indexes = translator.translate_sentence(src_tensor)

    trg_itos = trg_vocab.get_itos()

    predict_word = [trg_itos[index] for index in translated_indexes]
    print("I am a student . -> ", end="")
    print("".join(predict_word))  # 打印出来
