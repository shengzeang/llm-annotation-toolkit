import json
import torch
import gurobipy as gp
from gurobipy import GRB
from lxml.includes.xpath import xmlXPathEvalExpression

from annotation_example import num_classes
from ..utils import count_tokens, calculate_ranking_diff, query_oracle, get_embeddings_from_llm, query_oracle_for_psample


class ActiveLearning:
    def __init__(self, data, available_idx, budget):
        self._data = data
        self._num_nodes = data.x.shape[0]
        self._available_idx = available_idx
        self._budget = budget
        self._scores = torch.zeros(self._num_nodes) - 1

        self._preprocessing()

    def _preprocessing(self):
        return NotImplementedError

    def _score_calculation(self, cur_cnt, train_idx):
        return NotImplementedError

    def select_node(self, cur_cnt, train_idx):
        self._score_calculation(cur_cnt, train_idx)
        return self._scores.argmax().item()

    def update(self, selected_node):
        self._available_idx = self._available_idx[self._available_idx != selected_node]
        self._scores[selected_node] = -1


# later add recommendation, detection, and adjustment
class PromptSelection:
    def __init__(self, data, budget, token2money, num_iter, prompt_init_file_path="./prompts/default_prompts.json", adaptation=False, adaptation_budget=None, difficulty_score=None):
        # pre-defined indicators for sample descriptions
        # can be found in corresponding locations in datasets
        self._sample_indicators = ["CONTENT", "ONEHOP", "TWOHOP"]
        self._task_indicators = ["ENTITY", "DOMAIN", "CATEGORY", "DESCRIPTION", "DEMONSTRATION"]

        self._adaptation = False
        # if adaptation:
        #     assert(False)
        
        self._data = data
        # for preliminary test only
        self._budget = budget
        self._token2money = token2money
        self._num_iter = num_iter
        # load initial prompt template from json file
        self._load_prompt_from_json(prompt_init_file_path)
        # initialize accs to a set of recommended values
        self._prompt_accs, self._prompt_ranking = self._prompts_initilization()

        if adaptation:
            self._adaptation = True
            self._first_round_adaptation = True
            if adaptation_budget is None:
                assert False, 'Adaptation budget should be specified.'
            self._adaptation_budget = adaptation_budget
            self._ranking_diff_threshold = (
                len(self._prompts)**2 - len(self._prompts) % 2) // 4
            # detect if distribution shift exists
            # query the oracle to generate pseudo samples according to class descriptions
            self._global_noise_matrix = query_oracle_for_psample(self._data)
            self._local_noise_matrix = []
            self._embedding_cache = None
            self._class_assigns_cache = None

        if difficulty_score != None:
            self._difficulty_score = difficulty_score


    def _load_prompt_from_json(self, json_path):
        with open(json_path, "r") as json_file:
            prompts = json.load(json_file)
            # every prompt is composed of (required)sample_description, (required)task_description, 
            # (optional None)extra_sample_description, (optional None)extra_task_description
            # only special indicator in the prompt for later filled dataset-specific information
            self._prompts = []
            for _, prompt_template in prompts.items():
                # concatenate consecutively
                prompt = prompt_template["sample description"]
                if prompt_template["extra sample description"] != None:
                    prompt += prompt_template["extra sample description"]
                if prompt_template["extra task description"] != None:
                    prompt += prompt_template["extra task description"]
                prompt += prompt_template["task description"]
                self._prompts.append(prompt)
            self.num_prompts = len(self._prompts)


    def _prompts_initilization(self):
        # initialize with given recommended values
        # also initialize from files? sounds reasonable, currently only naive values
        # self._prompt_accs = torch.ones(len(self._prompts))
        # self._prompt_accs = torch.rand(len(self._prompts))
        self._prompt_accs = torch.tensor([1.0, 0.0])
        return self._prompt_accs, self._prompt_accs.argsort(descending=True)

    def _solve_ILP(self, prompt_cost, budget):
        solution = torch.zeros(prompt_cost.shape[0])
        m = gp.Model("0-1 ILP for prompt selection")
        m.setParam('OutputFlag', 0)

        # create variables
        choice_matrix = []
        for i in range(prompt_cost.shape[0]):
            choice_matrix.append([])
            for j in range(prompt_cost.shape[1]):
                choice_matrix[i].append(
                    m.addVar(vtype=GRB.BINARY, name=f"{i},{j}"))

        # set objective
        m.setObjective(gp.quicksum(choice_matrix[i][j] * self._prompt_accs[j]
                                   for i in range(prompt_cost.shape[0]) for j in range(prompt_cost.shape[1])), GRB.MAXIMIZE)

        # add constraints
        for i in range(prompt_cost.shape[0]):
            m.addConstr(gp.quicksum(choice_matrix[i][j]
                        for j in range(prompt_cost.shape[1])) == 1)
        m.addConstr(gp.quicksum(choice_matrix[i][j] * prompt_cost[i, j]
                                for i in range(prompt_cost.shape[0]) for j in range(prompt_cost.shape[1])) <= budget)

        # optimize model
        m.optimize()

        for v in m.getVars():
            if v.X == 1:
                node_idx, prompt_idx = [int(id) for id in v.VarName.split(',')]
                solution[node_idx] = prompt_idx
        return solution

    
    def gen_real_prompt(self, node_id, prompt_id):
        prompt = self._prompts[prompt_id].copy()
        while True:
            flag = False
            for indicator in self._sample_indicators + self._task_indicators:
                if indicator in prompt:
                    flag = True
                    pos = prompt.index(indicator)
                    # currently only ENTITY, DOMAIN, CATEGORY, DESCRIPTION
                    if indicator == "CONTENT":
                        prompt[pos] = self._data.raw_texts[node_id]
                    elif indicator == "ENTITY":
                        prompt[pos] = self._data.entity
                    elif indicator == "DOMAIN":
                        prompt[pos] = self._data.domain
                    elif indicator == "CATEGORY":
                        prompt[pos] = ", ".join(self._data.category_names)
                    elif indicator == "DESCRIPTION":
                        category_descriptions = self._data.category_descriptions
                        prompt[pos] = " ".join(category_descriptions.values())
            if flag == False:
                break
        return "".join(prompt)

    
    def select_prompt(self, prompt_token_lengths):
        # solve the ILP problem with Gurobi
        # later might change to more flexible budget assignment plan among different batches
        cur_batch_budget = self._budget // self._num_iter
        prompt_cost = prompt_token_lengths * self._token2money

        # determine proper prompt for each data sample
        solution = self._solve_ILP(prompt_cost, cur_batch_budget)
        return solution


    def get_annotations(self, data, node_list):
        prompt_token_lengths = torch.zeros(
            (len(node_list), len(self._prompts)))
        for i, selected_node in enumerate(node_list):
            for j, _ in enumerate(self._prompts):
                real_prompt = self.gen_real_prompt(selected_node, j)
                prompt_token_lengths[i, j] = count_tokens(real_prompt)

        if self._adaptation and self._adaptation_budget > 0:
            # generate embeddings for the node corresponding raw texts
            embeddings = get_embeddings_from_llm(data, node_list)

            # report error if adaptation budget cannot even support one round
            cur_round_cost = prompt_token_lengths.sum() * self._token2money
            if self._first_round_adaptation:
                self._first_round_adaptation = False
                assert self._adaptation_budget >= 2 * cur_round_cost, \
                    "Adaptation budget is too small. Cannot even sustain for two rounds."
            # query LLM for annotations of these nodes
            class_assigns = []
            for i in range(self.num_prompts):
                solution = [i] * len(node_list)
                class_assign = []
                # query llm according to certain prompt template
                for j, node in enumerate(node_list):
                    # construct prompt
                    prompt = self.gen_real_prompt(node, int(solution[j]))
                    # query llm
                    class_assign = query_oracle(data, prompt)
                class_assigns.append(class_assign)

            # ensemble to get final class assignment for each node during adaptation
            output_y = []
            class_assigns_mat_T = torch.tensor(class_assigns).transpose(0,1)
            for i in range(len(node_list)):
                elem_count = torch.bincount(class_assigns_mat_T[i], minlength=len(self._data.category_names))
                ensemble_assign = torch.argmax(elem_count).item()
                output_y.append(ensemble_assign)

            # store embeddings and class assignments for each prompt
            if self._embedding_cache is None:
                self._embedding_cache = embeddings
                self._class_assigns_cache = torch.tensor(class_assigns)
            else:
                self._embedding_cache = torch.cat(
                    [self._embedding_cache, embeddings], dim=0)
                self._class_assigns_cache = torch.cat(
                    [self._class_assigns_cache, torch.tensor(class_assigns)], dim=1)

            # update adaptation budget, and see if continue adaptation in the next round
            self._adaptation_budget -= cur_round_cost
            if self._adaptation_budget <= 0:
                # budget exhausted, initiate detection
                # each prompt generates its own noise matrix according to currently annotated samples by itself
                for k in range(self.num_prompts):
                    local_noise_matrix = torch.zeros((num_classes, num_classes))
                    # transform from sample->label to class->sample
                    class_to_sample = []
                    for _ in range(num_classes):
                        class_to_sample.append([])
                    has_class = torch.zeros(num_classes)
                    for sample_id,class_id in enumerate(self._class_assigns_cache[k]):
                        class_to_sample[class_id].append(sample_id)
                        has_class[class_id] = 1
                    # compute class-level representations
                    avail_class_id = []
                    avail_class_embedding = []
                    for i in range(num_classes):
                        if has_class[class_id] == 1:
                            avail_class_id.append(i)
                            avail_class_embedding.append(self._embedding_cache[class_to_sample[i]].mean(dim=0))
                    avail_class_embedding = torch.stack(avail_class_embedding)

                    # compute similarity value within each available class
                    # same class, compute pair-wise similarity, then average
                    for cla in avail_class_id:
                        num_samples = len(class_to_sample[cla])
                        for i in range(num_samples):
                            for j in range(i+1, num_samples):
                                sample_i = class_to_sample[cla][i]
                                sample_j = class_to_sample[cla][j]
                                cosine_similarity = self._embedding_cache[sample_i].dot(self._embedding_cache[sample_j]) / (
                                    self._embedding_cache[sample_i].norm() * self._embedding_cache[sample_j].norm())
                                local_noise_matrix[cla, cla] += cosine_similarity
                        local_noise_matrix[cla, cla] /= num_samples*(num_samples-1)/2

                    # compute local noise matrix
                    for i, ii in enumerate(avail_class_id):
                        for j, jj in enumerate(avail_class_id):
                            if i == j: continue
                            # different classes, compute class-level representation, then calculate similarity
                            cosine_similarity = avail_class_embedding[i].dot(avail_class_embedding[j]) / (
                                    avail_class_embedding[i].norm() * avail_class_embedding[j].norm())
                            local_noise_matrix[ii, jj] = cosine_similarity
                            local_noise_matrix[jj, ii] = cosine_similarity

                    self._local_noise_matrix.append(local_noise_matrix)

                # each prompt calculates its relative difference, then ascendingly sort
                difference_list = []
                for local_noise_matrix in self._local_noise_matrix:
                    nz_pos = local_noise_matrix != 0
                    difference_list.append((local_noise_matrix[nz_pos] - self._global_noise_matrix[nz_pos]).abs().sum().item())
                # calculate ranking difference
                current_ranking = torch.tensor(difference_list).argsort(descending=True)
                if calculate_ranking_diff(self._prompt_ranking, current_ranking) > self._ranking_diff_threshold:
                    # if the ranking changes too much, then adjust prompt accuracies
                    print("Distribution shift detected! Adjusting prompt accuracies...")
                    # adjust
                    # xxx

        else:
            output_y = []
            solution = self.select_prompt(prompt_token_lengths)
            # query llm according to certain prompt template
            for j, node in enumerate(node_list):
                # construct prompt
                prompt = self.gen_real_prompt(node, int(solution[j]))
                # query llm
                class_assign = query_oracle(data, prompt)
                output_y.append(class_assign)

        return torch.tensor(output_y)


class DifficultyCalculation:
    def __init__(self):
        pass
