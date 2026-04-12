"""
utils.py
========
Utility functions: logging setup, data loading, DataLoader building,
and model saving.
"""

import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import config as config_hp
from src.dataset import HateSpeechDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# NEW — load config.py directly from your local config folder
import importlib.util

def load_config() -> dict:
    """Load config dict directly from config/config.py."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),  # project root
        'config', 'config.py'
    )
    spec   = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = module.config()
    logger.info("Config loaded from: %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> None:
    """Add a file handler to the root logger for the given output dir."""
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, 'training.log')
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to: %s", os.path.join(output_dir, 'training.log'))


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU   : %s", torch.cuda.get_device_name(0))
    return device


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config: dict) -> tuple:
    """
    Load train, test and annotated DataFrames from paths in config.
    Returns (df_train, df_test, df_annotated).
    """
    paths = config['PATHS']

    df_train     = pd.read_csv(paths['train_path'])
    df_test      = pd.read_csv(paths['test_path'])
    df_annotated = pd.read_csv(paths['annotated_path'])

    label_col = paths['label_col']

    logger.info("Train     : %d rows | pos=%d neg=%d",
                len(df_train),
                (df_train[label_col] == 1).sum(),
                (df_train[label_col] == 0).sum())
    logger.info("Test      : %d rows | pos=%d neg=%d",
                len(df_test),
                (df_test[label_col] == 1).sum(),
                (df_test[label_col] == 0).sum())
    logger.info("Annotated : %d rows | pos=%d neg=%d",
                len(df_annotated),
                (df_annotated[label_col] == 1).sum(),
                (df_annotated[label_col] == 0).sum())

    return df_train, df_test, df_annotated


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(config: dict) -> BertTokenizer:
    """Load BertTokenizer from config model name."""
    model_name = config['BERT']['model_name']
    logger.info("Loading tokenizer: %s", model_name)
    return BertTokenizer.from_pretrained(model_name)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    df_train:     pd.DataFrame,
    df_test:      pd.DataFrame,
    df_annotated: pd.DataFrame,
    tokenizer:    BertTokenizer,
    config:       dict,
    batch_size:   int,
) -> tuple:
    """
    Build and return (train_loader, test_loader, annotated_loader).
    """
    paths    = config['PATHS']
    text_col = paths['text_col']
    lbl_col  = paths['label_col']
    max_len  = config['BERT']['max_seq_length']

    train_dataset = HateSpeechDataset(
        df_train, tokenizer, text_col, lbl_col, max_len
    )
    test_dataset = HateSpeechDataset(
        df_test, tokenizer, text_col, lbl_col, max_len
    )
    annotated_dataset = HateSpeechDataset(
        df_annotated, tokenizer, text_col, lbl_col, max_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    annotated_loader = DataLoader(
        annotated_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    return train_loader, test_loader, annotated_loader


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_checkpoint(model, tokenizer, path: str) -> None:
    """Save model and tokenizer to path."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info("Checkpoint saved: %s", path)
