import torch
from datasets import load_dataset
from src.modules.al_modules import RANDOM, AGE

class QSAnnotator:
    def __init__(self, llm, tokenizer, max_tokens=512, batch_size=1):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.batch_size = batch_size
    
    def annotate_dataset(self, data, budget, token2money, active_learning, verbose=False):
        aval_idx = torch.arange(len(data))
        train_idx = []

        # certain active learning algorithm to annotate the dataset
        # input: dataset, budget, specific algorithm; output: the annotated dataset
        node_selection = self.gen_active_learning_from_name(active_learning, data, aval_idx, budget)
        batch_size = self.batch_size
        num_iter = (budget + batch_size - 1) // batch_size

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
            # adaptation or not
            data[batch_selected_node_list]['answers'] = self.get_annotations(data, batch_selected_node_list, budget, token2money, num_iter=num_iter)

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.test_mask = ~data.train_mask
        return data

    def gen_active_learning_from_name(self, name, data, available_idx, budget):
        if name == 'RANDOM':
            return RANDOM(data, available_idx, budget)
        elif name == 'AGE':
            return AGE(data, available_idx, budget)
        else:
            raise NotImplementedError
        
    def get_annotations(self, data, node_list, budget, token2money, num_iter):
        #简略的标注过程,暂未考虑budget等过程,仅展示对于QA任务如何使用llm进行标注
        ## 对于不同任务的标注过程,需要根据具体情况进行修改
        annotations = []
        for i in range(len(node_list)):
            node = node_list[i]
            context = data[node]['context']
            question = data[node]['question']
            prompt = self.create_prompt(context, question)

            ## 也许可以在数据集data里加入一个list,用于存储在这一步需要的信息
            ## 比如说用data.components存储每个样本包含的组件
            ## prompt = self.create_prompt({component :data[node][component] for component in components})
            ## 然后在create_prompt里用统一的模板添加这个dict

            answer = self.query_llm(prompt)
            annotations.append(answer)
        return annotations

    def create_prompt(self, context, question):
        ## 不同任务的prompt构建方式不同
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return prompt

    def query_llm(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_tokens, truncation=True)
        outputs = self.llm.generate(**inputs, max_new_tokens=50, eos_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


# 使用示例
# from transformers import AutoModelForCausalLM, AutoTokenizer

# llm = AutoModelForCausalLM.from_pretrained("gpt-3")
# tokenizer = AutoTokenizer.from_pretrained("gpt-3")

# annotator = QSAnnotator(llm, tokenizer)
# dataset = load_dataset("squad")
# data = dataset["train"]
# annotations = annotator.annotate_dataset(data, 100, token2money, "RANDOM", verbose=True)
