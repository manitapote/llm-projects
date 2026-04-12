"""
train.py
========
Entry point for a single BERT training run
using fixed hyperparameters from config.py.
"""

import logging
import os
from datetime import datetime

import pandas as pd
import torch

from src.metrics  import log_metrics
from src.model    import build_model, build_optimizer, build_scheduler, build_loss
from src.trainer  import train_epoch, evaluate
from src.utils    import (
    load_config, setup_logging, get_device,
    load_data, load_tokenizer, build_dataloaders, save_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_training(config: dict) -> None:
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['PATHS']['output_dir'], f'run_{timestamp}')
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    setup_logging(output_dir)

    device                          = get_device()
    df_train, df_test, df_annotated = load_data(config)
    tokenizer                       = load_tokenizer(config)

    bert_cfg   = config['BERT']
    lr         = bert_cfg['learning_rate'][0]   # use first lr for single run
    epochs     = bert_cfg['stage1_epochs']
    batch_size = bert_cfg['stage1_batch_size']

    train_loader, test_loader, annotated_loader = build_dataloaders(
        df_train, df_test, df_annotated,
        tokenizer, config, batch_size,
    )

    model     = build_model(config, device)
    loss_fn   = build_loss(df_train, config, device)
    optimizer = build_optimizer(model, config, lr)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(bert_cfg['warmup_ratio'] * total_steps)
    scheduler    = build_scheduler(optimizer, total_steps, warmup_steps)

    logger.info("lr=%.0e | epochs=%d | batch=%d | steps=%d | warmup=%d",
                lr, epochs, batch_size, total_steps, warmup_steps)

    history  = []
    best_f1  = 0.0

    for epoch in range(1, epochs + 1):
        logger.info("=" * 50)
        logger.info("EPOCH %d / %d", epoch, epochs)

        train_loss                    = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
        test_loss,  test_metrics      = evaluate(model, test_loader,      loss_fn, device, "test")
        ann_loss,   ann_metrics       = evaluate(model, annotated_loader,  loss_fn, device, "annotated")

        logger.info("Train: %.4f | Test: %.4f | Gap: %.4f",
                    train_loss, test_loss, test_loss - train_loss)
        log_metrics(test_metrics)
        log_metrics(ann_metrics)

        history.append({
            'epoch':         epoch,
            'train_loss':    train_loss,
            'test_loss':     test_loss,
            'test_f1_hate':  test_metrics['f1_hate'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_accuracy': test_metrics['accuracy'],
            'ann_f1_hate':   ann_metrics['f1_hate'],
        })

        # Save checkpoint
        ckpt = os.path.join(output_dir, 'checkpoints', f'epoch_{epoch}')
        save_checkpoint(model, tokenizer, ckpt)

        # Save best model
        if test_metrics['f1_hate'] > best_f1:
            best_f1 = test_metrics['f1_hate']
            save_checkpoint(model, tokenizer, os.path.join(output_dir, 'best_model'))
            logger.info("New best model: F1=%.4f", best_f1)

    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(output_dir, 'history.csv'), index=False)
    logger.info("History saved. Best F1: %.4f", best_f1)
    print(df_history.to_string(index=False))


if __name__ == "__main__":
    config = load_config()
    run_training(config)
