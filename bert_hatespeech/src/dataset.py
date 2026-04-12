"""
dataset.py
==========
PyTorch Dataset for hate speech classification.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer


class HateSpeechDataset(Dataset):
    """
    Tokenizes text and returns input tensors for BERT.

    Parameters
    ----------
    df        : DataFrame with text and label columns
    tokenizer : BertTokenizer instance
    text_col  : column name for input text
    label_col : column name for binary label (0/1)
    max_len   : maximum token sequence length
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        tokenizer: BertTokenizer,
        text_col:  str,
        label_col: str,
        max_len:   int,
    ):
        self.texts     = df[text_col].astype(str).tolist()
        self.labels    = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length     = self.max_len,
            padding        = 'max_length',
            truncation     = True,
            return_tensors = 'pt',
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }
