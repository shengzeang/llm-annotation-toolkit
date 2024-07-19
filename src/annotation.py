import torch

from .modules.base_modules import PromptSelection
from .modules.al_modules import RANDOM, AGE
from .utils import query_oracle


def annotate_dataset_pyg(data, budget, token2money, active_learning, verbose=False):
    x, y, edge_index = data.x, data.y, data.edge_index
    train_idx = []
    aval_idx = torch.arange(data.num_nodes)

    # certain active learning algorithm to annotate the dataset
    # input: dataset, budget, specific algorithm; output: the annotated dataset
    num_classes = data.y.max().item() + 1
    node_selection = gen_active_learning_from_name(active_learning, data, aval_idx, budget)
    batch_size = num_classes
    num_iter = (budget + batch_size - 1) // batch_size
    prompt_selection = PromptSelection(data, budget, token2money, num_iter=num_iter)

    cur_cnt = 0
    for i in range(num_iter):
        batch_selected_node_list = []
        for _ in range(batch_size):
            if verbose:
                print(f'Selecting {cur_cnt}th sample...')
            selected_node = node_selection.select_node(cur_cnt, train_idx)
            train_idx.append(selected_node)
            batch_selected_node_list.append(selected_node)

            node_selection.update(selected_node)

            cur_cnt += 1
            if cur_cnt >= budget:
                break

        if verbose:
            print(f'Annotating {num_iter}th batch...')
        y[batch_selected_node_list] = query_oracle(data, batch_selected_node_list, prompt_selection)

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask = ~data.train_mask
    return data


def gen_active_learning_from_name(name, data, available_idx, budget):
    if name == 'RANDOM':
        return RANDOM(data, available_idx, budget)
    elif name == 'AGE':
        return AGE(data, available_idx, budget)
    else:
        raise NotImplementedError
