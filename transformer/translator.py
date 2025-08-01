''' This module will handle the text generation with beam search. '''

import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import Transformer
from models.transformer import get_src_mask
from models.transformer import get_trg_mask

class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_trg_mask(trg_seq, self.model.trg_pad_idx)


        dec_output = self.model.decoder(trg_seq, enc_output, trg_mask, src_mask)
        #return F.softmax(self.model.fc(dec_output), dim=-1)
        return F.softmax(dec_output, dim=-1)

    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.beam_size

        enc_output = self.model.encoder(src_seq, src_mask)

        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            src_mask = get_src_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()

from dataset import TranslateDataset
from dataset import build_vocab

if __name__ == '__main__':
    # 使用TranslateDataset读取训练数据，得到数据集dataset
    dataset = TranslateDataset("./data/small.txt")

    # 使用build_vocab，建立源语言词表和目标语言词表
    src_vocab, trg_vocab = build_vocab(dataset)
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

    beam_size = 5
    max_seq_len = 16

    translator = Translator(model,
                            beam_size,
                            max_seq_len,
                            src_pad_idx,
                            trg_pad_idx,
                            trg_bos_idx,
                            trg_eos_idx).to(DEVICE)

    src_tokens, _ = dataset[0]

    # 将源句子转换为索引形式
    src_index = [src_vocab[token] for token in src_tokens]

    src_tensor = torch.LongTensor(src_index).unsqueeze(0).to(DEVICE)

    # 使用translator进行翻译
    translated_indexes = translator.translate_sentence(src_tensor)

    trg_itos = trg_vocab.get_itos()

    # 将索引转换回文本形式
    predict_word = [trg_itos[index] for index in translated_indexes]

    # 打印翻译结果
    print("原句:", ' '.join(src_tokens))
    print("翻译:", ' '.join(predict_word))

