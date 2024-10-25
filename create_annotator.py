import torch

from nlp_ann_class import NLPAnnotator
from transformers import AutoModelForCausalLM, AutoTokenizer

llm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")

# Natural language inference
class snliAnnotator(NLPAnnotator):
    def create_prompt(self, node):
        premise = node['premise']
        hypothesis = node['hypothesis']
        prompt = "premise: A man inspects the uniform of a figure in some East Asian country.\nhypothesis: The man is sleeping.\nlabel: contradiction\n\n"
        +"premise: An older and younger man smiling.\nhypothesis: Two men are smiling and laughing at the cats playing on the floor.\nlabel: neutral\n\n"
        +"premise: A soccer game with multiple males playing.\nhypothesis: Some men are playing a sport.\nlabel: entailment\n\n"
        +f"premise: {premise}\nhypothesis: {hypothesis}\nlabel:"
        return prompt

# Question answering
class squadAnnotator(NLPAnnotator):
    def create_prompt(self, node):
        context = node['context']
        question = node['question']
        title = node['title']
        prompt = "Title: University_of_Notre_Dame\nContext: Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.\nQuestion: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?\n"
        +"Answers: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}"
        +f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswers:"
        return prompt


# Sentiment Analysis
class sstannotator(NLPAnnotator):
    def create_prompt(self, node):
        text = node['text']
        label = node['label']
        prompt = "analysis the sentiment of the following text, output the classification result. 0 for very negative, 1 for negative, 2 for neutral, 3 for positive, 4 for very positive\n"
        +f"text: {text}\nlabel:"
        return prompt
    
# Common Sense Reasoning
class CommonsenseQAAnnotator(NLPAnnotator):
    def create_prompt(self, node):
        question = node['question']
        question_concept = node['question_concept']
        choises = node['choises']
        prompt = "Given the following question and the concept, select the most reasonable answer\n"
        +"question: What home entertainment equipment requires cable?\n"
        +"question_concept: cable\n"
        +"choises: { label: [ A, B, C, D, E ], text: [ radio shack, substation, cabinet, television, desk ] }\n"
        +"answerKey: D\n\n"
        +f"question: {question}\nquestion_concept: {question_concept}\nchoises: {choises}\nanswerKey:"
        return prompt


snli_annotator = snliAnnotator(llm = llm, tokenizer = tokenizer)
squad_annotator = squadAnnotator(llm = llm, tokenizer = tokenizer)
sst_annotator = sstannotator(llm = llm, tokenizer = tokenizer)
commonsenseqa_annotator = CommonsenseQAAnnotator(llm = llm, tokenizer = tokenizer)