"""
model.py
========
Model, optimizer, scheduler and loss function setup for BERT.
"""

import logging
import numpy as np
import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

logger = logging.getLogger(__name__)


def build_model(config: dict, device: torch.device) -> BertForSequenceClassification:
    """Load pretrained BERT and move to device."""
    logger.info("Loading model: %s", config['BERT']['model_name'])
    model = BertForSequenceClassification.from_pretrained(
        config['BERT']['model_name'],
        num_labels                   = config['BERT']['num_labels'],
        hidden_dropout_prob          = config['BERT']['dropout'],
        attention_probs_dropout_prob = config['BERT']['dropout'],
    )
    model.to(device)
    return model


def build_optimizer(model: BertForSequenceClassification, config: dict, learning_rate: float) -> AdamW:
    """
    AdamW optimizer with weight decay applied to weights only.
    Bias and LayerNorm parameters are excluded from weight decay.
    """
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_params = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': config['BERT']['weight_decay'],
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    return AdamW(
        grouped_params,
        lr  = learning_rate,
        eps = config['BERT']['adam_epsilon'],
    )


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup + linear decay scheduler."""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )


def build_loss(
    df_train: pd.DataFrame,
    config:   dict,
    device:   torch.device,
) -> nn.CrossEntropyLoss:
    """
    CrossEntropyLoss with optional class weights for imbalanced data.
    Weights computed from training label distribution.
    """
    if config['BERT']['use_class_weights']:
        weights = compute_class_weight(
            class_weight = 'balanced',
            classes      = np.array([0, 1]),
            y            = df_train[config['PATHS']['label_col']].astype(int).values,
        )
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        logger.info("Class weights: %s", weights)
    else:
        class_weights = None

    return nn.CrossEntropyLoss(weight=class_weights)
