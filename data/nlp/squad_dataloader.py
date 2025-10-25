from datasets import load_dataset
from torch.utils.data import Dataset
from argparse import ArgumentParser
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json


def filter(x):
    return x['answers']['text'] != []


class SQuADDataset(Dataset):
    def __init__(self, hparams, split):
        self.hparams = hparams
        hf_token = os.getenv('HF_TOKEN')
        hf_home = os.getenv('HF_HOME')
        dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else hf_home
        self.dataset = load_dataset("squad_v2", cache_dir=dataset_dir, token=hf_token)[split]
        self.dataset = self.dataset.filter(filter)

        # Filter examples by length for finetune mode to ensure [[Answer]]: tag is not truncated
        if self.hparams.execution_mode == "finetune":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces=False)

            def length_filter(example):
                text = f"[[Question]]: {example['context']}\n{example['question']}\n[[Answer]]: {example['answers']['text'][0]}"
                tokens = tokenizer.encode(text, add_special_tokens=False)
                # Allow +1 for next token prediction (context_length + 1 in collator)
                return len(tokens) <= self.hparams.context_length

            original_len = len(self.dataset)
            self.dataset = self.dataset.filter(length_filter)
            filtered_len = len(self.dataset)
            if split == "train":
                print(f"Filtered {original_len - filtered_len} examples (>{self.hparams.context_length} tokens). Remaining: {filtered_len}")
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.hparams.execution_mode == "inference":
            return f"[[Question]]: {self.dataset[idx]['context']}\n{self.dataset[idx]['question']}\n[[Answer]]: ", self.dataset[idx]['answers']['text'][0]
        elif self.hparams.execution_mode == "pretrain":
            return f"Question: {self.dataset[idx]['context']}\n{self.dataset[idx]['question']}\nAnswer: {self.dataset[idx]['answers']['text'][0]}"
        elif self.hparams.execution_mode == "finetune":
            return f"[[Question]]: {self.dataset[idx]['context']}\n{self.dataset[idx]['question']}\n[[Answer]]: {self.dataset[idx]['answers']['text'][0]}"
        else:
            raise ValueError(f"Execution mode not supported. Please add support for mode {self.hparams.execution_mode}")
    
    def __repr__(self, idx):
        return f"Context: {self.dataset[idx]['context']}\nQuestion: {self.dataset[idx]['question']}\nAnswer: {self.dataset[idx]['answer']}"