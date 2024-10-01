from datasets import load_dataset
import torch
import os.path as osp

class Anil:
    def __init__(self, root="./"):
        self.name = "anil"
        self.dataset = load_dataset("facebook/anli", cache_dir=root)

    
    @property
    def entity(self):
        return "Two paragraphs, namely, premise and hypothesis."
    
    @property
    def domain(self):
        return "Natural Language Inference"
    
    @property
    def raw_texts(self):
        return self.dataset
    
    @property
    def category_names(self):
        return [
            'entailment',
            'neutral',
            'contradiction'
        ]
    
    @property
    def category_descriptions(self):
        entailment = "The hypothesis can be inferred from the premise."
        neutral = "The hypothesis is neither supported nor refuted by the premise."
        contradiction = "The hypothesis is refuted by the premise."
        return {
            'entailment': entailment,
            'neutral': neutral,
            'contradiction': contradiction
        }