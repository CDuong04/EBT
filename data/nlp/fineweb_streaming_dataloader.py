from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset
import os

class FineWebStreamingDataset(IterableDataset):
    """Streaming version of FineWeb dataset - downloads on-the-fly, no storage needed"""

    def __init__(self, hparams):
        if hparams.execution_mode != "pretrain":
            raise ValueError("FineWeb is a pretrain dataset, no other execution modes supported.")

        self.max_length = hparams.context_length + 1
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces=False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        hf_home = os.getenv('HF_HOME')
        dataset_dir = hparams.dataset_dir if hparams.dataset_dir != "" else hf_home

        # Use sample-100BT for 10x more data with streaming (no download needed!)
        print(f"Loading {hparams.dataset_name} dataset in streaming mode (sample-100BT, ~100B tokens)")
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            "sample-100BT",
            split="train",
            cache_dir=dataset_dir,
            streaming=True  # Key: streaming mode!
        )

        # Shuffle with a large buffer for better randomness
        buffer_size = getattr(hparams, 'streaming_buffer_size', 10000)
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=buffer_size)

        self.hparams = hparams

    def tokenize_example(self, example):
        """Tokenize a single example on-the-fly"""
        tokenized = self.tokenizer(
            example['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0)
        }

    def __iter__(self):
        """Iterate through streaming dataset"""
        for example in self.dataset:
            yield self.tokenize_example(example)
