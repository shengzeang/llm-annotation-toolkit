import torch
import numpy as np
import tiktoken
import random

from torch_geometric.utils import to_torch_csc_tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from transformers import AutoModelForCausalLM, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("/data2/stansheng/mistral7binst/")
# model = AutoModelForCausalLM.from_pretrained("/data2/stansheng/mistral7binst/", device_map="balanced_low_0")


def query_oracle(data, node_list, prompt_selection):
    output_y = []
    solution = prompt_selection.select_prompt(node_list)
    # query llm according to certain prompt template
    for j, node in enumerate(node_list):
        # construct prompt, todo
        prompt = prompt_selection.gen_real_prompt(node, int(solution[j]))
        # print(prompt)
        # exit()

        # query llm
        '''inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=2, pad_token_id=2)
        classification = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")
        class_assign = -1
        num_classes = len(data.category_names)
        for i in range(num_classes):
            if data.categories[i] in classification:
                class_assign = i
                break
        if class_assign == -1:
            class_assign = random.randint(0, num_classes-1)'''
        class_assign = data.y[node]
        output_y.append(class_assign)

    return torch.tensor(output_y)


def count_tokens(text, encoding='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def calculate_ranking_diff(a_rank, b_rank):
    sum_diff = 0
    for i, a in enumerate(a_rank):
        for j, b in enumerate(b_rank):
            if a == b:
                sum_diff += abs(i-j)
    return sum_diff


def feature_propagation(data, num_hops):
    edge_weight = data.edge_attr
    if 'edge_weight' in data:
        edge_weight = data.edge_weight
    adj_t = to_torch_csc_tensor(
                edge_index=data.edge_index,
                edge_attr=edge_weight,
                size=data.size(0),
            ).t()
    adj_t, _ = gcn_norm(adj_t, add_self_loops=False)

    out = data.x.clone()
    for _ in range(num_hops):
        out = adj_t @ out
    return out
