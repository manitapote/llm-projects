"""
metrics.py
==========
Metric computation and logging for hate speech classification.
"""

import logging
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: list, y_pred: list, split_name: str) -> dict:
    """
    Compute classification metrics for binary hate speech detection.

    Returns a dict with accuracy, F1 macro, F1/precision/recall
    for the positive (hate speech) class, and a full sklearn report.
    """
    return {
        'split':     split_name,
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1_macro':  f1_score(y_true, y_pred, average='macro',  zero_division=0),
        'f1_hate':   f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        'precision': precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        'recall':    recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        'report':    classification_report(
                         y_true, y_pred,
                         target_names=['control', 'hatespeech'],
                         output_dict=True,
                         zero_division=0,
                     ),
    }


def log_metrics(metrics: dict) -> None:
    """Log metrics to the logger."""
    logger.info("  Split      : %s", metrics['split'])
    logger.info("  Accuracy   : %.4f", metrics['accuracy'])
    logger.info("  F1 macro   : %.4f", metrics['f1_macro'])
    logger.info("  F1 hate    : %.4f", metrics['f1_hate'])
    logger.info("  Precision  : %.4f", metrics['precision'])
    logger.info("  Recall     : %.4f", metrics['recall'])
