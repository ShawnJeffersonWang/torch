import torch
from torch.nn import functional as F
from load_gpt2_model import load_gpt2_model


def generate_random(model, idx, max_new_tokens):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :]
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_greedy(model, idx, max_new_tokens):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, using a greedy decoding approach.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long, we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits = model(idx_cond)
        # pluck the logits at the final step and select the highest probability index
        idx_next = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # append the highest probability index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == '__main__':
    config_json = 'openai-gpt2/config.json'
    pytorch_model_bin = 'openai-gpt2/pytorch_model.bin'
    gpt2 = load_gpt2_model(config_json, pytorch_model_bin)
    print("gpt2:")
    print(gpt2)
    gpt2.eval()

    # 某种初始化的序列索引，通常是编码过的
    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    # 生成10个新的词
    generated_idx = generate_random(gpt2, idx, max_new_tokens=10)
    # 打印输出序列
    print("Generated indices:", generated_idx)

    # 生成10个新的词
    generated_idx = generate_greedy(gpt2, idx, max_new_tokens=10)
    # 打印输出序列
    print("Generated indices:", generated_idx)
