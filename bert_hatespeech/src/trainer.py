"""
trainer.py
==========
Training and evaluation loops for BERT hate speech detection.
"""

import logging
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import compute_metrics

logger = logging.getLogger(__name__)


def train_epoch(
    model,
    loader:    DataLoader,
    optimizer,
    scheduler,
    loss_fn:   nn.CrossEntropyLoss,
    device:    torch.device,
    epoch:     int,
) -> float:
    """
    Run one training epoch.
    Returns average loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logger.info(
                "  Epoch %d | Batch %d/%d | Loss: %.4f",
                epoch, batch_idx, len(loader), loss.item(),
            )

    return total_loss / len(loader)


def evaluate(
    model,
    loader:     DataLoader,
    loss_fn:    nn.CrossEntropyLoss,
    device:     torch.device,
    split_name: str,
) -> tuple:
    """
    Run evaluation on a DataLoader.
    Returns (avg_loss, metrics_dict).
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
            )

            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds, split_name)
    return avg_loss, metrics
