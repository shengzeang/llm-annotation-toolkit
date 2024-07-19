# llm-annotation-toolkit

This repository contains an annotation toolkit with the help of LLMs. This toolkit is under development. Thus, many planned functionalities are not properly implemented currently.



### Draft architecture

<img src="https://github.com/shengzeang/llm-annotation-toolkit/blob/main/draft_architecture.png" style="zoom:50%;" />

**Three main modules (under development)**:

1. **Difficulty Score Calculation Module** (Optional): apply data type/domain specific algorithms to assign difficulty scores to samples in the dataset, which reflect annotation difficulty; these sample-wise difficulty scores are later used in **Prompt Template Selection Module**.
2. **Active Learning Module**: contains various data type/domain specific active learning algorithms that select valuable data samples and send them to **Prompt Template Selection Module** for annotation.
3. **Prompt Template Selection Module**: contains various prompt templates and each of these templates is associated with an approximate performance score (recommend value) that is produced from evaluation on many pre-existed datasets; this module detects whether these recommended values is proper for the input dataset and adjusts these performance scores for prompt templates.



### Main features

#### Flexible:

1. **Easy for extension**: easy to incorporate new active learning algorithms and prompt templates under the unified abstraction;
2. **Data type/domain aware**: can switch between different algorithms inside each module according to data types/domains (e.g., different active learning algorithms for graph and tabular data).

#### Monetary cost reduction focused:

1. **More fine-grained cost model**: the first to model LLM annotation cost at the token-level;
2. **Annotation difficulty aware**: assign less token budget to easy data samples, reducing unnecessary monetary cost.



### Preliminary usage

This repository contains a preliminary example, *annotation_example.py*, that annotates the unlabeled Cora dataset and then train a 2-layer GCN on the annotated dataset. 

Compared to the original *gcn_pyg.py* from PyG's official example, we only make two modifications to the code to enable LLM annotation:

Firstly, use our pre-processed version of the famous planetoid dataset, *RawPlanetoid*. The major modification is to add raw text descriptions to the original PyG dataset.

```python
from src.dataset import RawPlanetoid
dataset = RawPlanetoid(path, args.dataset)
```

Secondly, use *annotate_dataset_pyg* to annotate the dataset, the return *data* object contains new *train_mask* and the training nodes are annotated with LLM annotated labels.

```python
from src.annotation import annotate_dataset_pyg
data = annotate_dataset_pyg(data, budget=num_classes*20, token2money=0.001, active_learning='RANDOM')
```



**Brief instruction for annotating node-level datasets in PyG**:

1. copy *planetoid.py* under the path of *src/dataset*, and rename the copied file to any desired name;
2. create a new class and make it inherit the desired dataset in PyG (e.g., amazon);
3. change the property methods in the original *RawPlanetoid* class to new values for the new dataset;
4. prepare the raw text descriptions for each node, and store it as a dict in a *.pt* file, keyed by *raw_texts*, and load using *torch.load*, substitute the original *self.\_raw\_*data;
5. change the "PATH_TO_LLM" to your path for deployed LLM in src/utils.py;
6. start annotation with *annotate_dataset_pyg* method, see *annotation_example.py* for  details.
