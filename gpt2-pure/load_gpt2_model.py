import torch
import json
from models.gpt2 import GPT2

def load_gpt2_model(config_json, pytorch_model_bin):
    with open(config_json, 'r') as file:
        config = json.load(file)

    vocab_size = config['vocab_size']
    block_size = config['n_ctx']
    n_embd = config['n_embd']
    n_head = config['n_head']
    n_layer = config['n_layer']
    drop_rate = 0

    model = GPT2(vocab_size, block_size, n_embd,
                 n_head, drop_rate, n_layer)

    weights = torch.load(pytorch_model_bin)

    keys_to_remove = ['h.0.attn.bias', 'h.1.attn.bias',
                      'h.2.attn.bias', 'h.3.attn.bias',
                      'h.4.attn.bias', 'h.5.attn.bias',
                      'h.6.attn.bias', 'h.7.attn.bias',
                      'h.8.attn.bias', 'h.9.attn.bias',
                      'h.10.attn.bias', 'h.11.attn.bias']

    for key in keys_to_remove:
        if key in weights:
            del weights[key]

    # 创建一个新的 OrderedDict 来存储更新后的键值对
    new_weights = {}

    # 遍历原始weights中的每个键值对，更新键的名称
    for key, value in weights.items():
        new_key = f"transformer.{key}"  # 在键前添加'transformer.'
        new_weights[new_key] = value

    # 添加新的键 lm_head.weight，并将其值设置为wpe.weight的副本
    new_weights['lm_head.weight'] = weights['wte.weight'].clone()

    model.load_state_dict(new_weights)
    return model

def compare_model_parameters(model1, model2):
    # 获取两个模型的状态字典
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    result = True

    # 检查键名是否一致
    if sd1.keys() != sd2.keys():
        print("模型的参数键名不匹配。")
        result = False

    # 逐一比较每个参数
    for key in sd1:
        if not torch.equal(sd1[key], sd2[key]):
            result = False
            print(f"参数 {key} 在两个模型中不相同。")
            print(f"形状：{sd1[key].shape} vs {sd2[key].shape}")
            if sd1[key].shape == sd2[key].shape:  # 只有形状相同时才进一步分析
                diff = sd1[key] - sd2[key]
                print(f"最大差异: {torch.max(diff)}")
                print(f"最小差异: {torch.min(diff)}")
                print(f"均值差异: {torch.mean(diff)}")
                print(f"标准差: {torch.std(diff)}")
    return result

def test_model_exact_equivalence(model1, model2, input_ids):
    # 确保两个模型处于评估模式
    model1.eval()
    model2.eval()
    outputs1 = model1(input_ids=input_ids, return_dict=True).logits
    outputs2 = model2(input_ids)
    print("output1 shape: ", outputs1.shape)
    print("output2 shape: ", outputs2.shape)
    # 检查两个输出是否完全相同
    return torch.equal(outputs1, outputs2)

from transformers import GPT2LMHeadModel

if __name__ == '__main__':
    model_path = 'openai-gpt2'
    hf_model = GPT2LMHeadModel.from_pretrained(model_path)
    print("hf_model:")
    print(hf_model)
    print("hf_model config:")
    print(hf_model.config)

    config_json = 'openai-gpt2/config.json'
    pytorch_model_bin = 'openai-gpt2/pytorch_model.bin'
    gpt2 = load_gpt2_model(config_json, pytorch_model_bin)
    print("gpt2:")
    print(gpt2)

    if compare_model_parameters(hf_model, gpt2):
        print("两个模型中的参数完全相同。")

    # 随机生成输入序列
    block_size = hf_model.config.n_ctx  # 假设hf_model和gpt2使用相同的block_size
    input_ids = torch.randint(low=0, high=hf_model.config.vocab_size, size=(1, block_size))

    # 测试模型是否完全等效
    equivalent = test_model_exact_equivalence(hf_model, gpt2, input_ids)
    print("两个模型输出是否完全等同:", equivalent)



